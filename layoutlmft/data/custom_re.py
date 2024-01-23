# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os

import datasets

from layoutlmft.data.image_utils import load_image, normalize_bbox

import numpy as np

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{Jaume2019FUNSDAD,
  title={FUNSD: A Dataset for Form Understanding in Noisy Scanned Documents},
  author={Guillaume Jaume and H. K. Ekenel and J. Thiran},
  journal={2019 International Conference on Document Analysis and Recognition Workshops (ICDARW)},
  year={2019},
  volume={2},
  pages={1-6}
}
"""

_DESCRIPTION = """\
https://guillaumejaume.github.io/FUNSD/
"""


class CustomDSReConfig(datasets.BuilderConfig):
    """BuilderConfig for our custom dataset"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CustomDSReConfig, self).__init__(**kwargs)


class CustomDS(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CustomDSReConfig(name="CustomDS", version=datasets.Version("1.0.0"), description="Custom dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "entities": datasets.Sequence(
                        {
                            "start": datasets.Value("int64"),
                            "end": datasets.Value("int64"),
                            "label": datasets.Value("int64"),
                        }
                    ),
                    "relations": datasets.Sequence(
                        {
                            "head": datasets.Value("int64"),
                            "tail": datasets.Value("int64"),
                        }
                    ),
                    "image": datasets.Array3D(shape=(3, 224, 224), dtype="uint8"),
                    "image_path": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://guillaumejaume.github.io/FUNSD/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        funsd = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        custom_ds = "./layoutlmft/data/datasets/"
        # print("funsd: ", funsd)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{custom_ds}/training_data/", "img_dir": f"{funsd}/dataset/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{custom_ds}/testing_data/", "img_dir": f"{funsd}/dataset/"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath, img_dir):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = filepath
        split = "training" if "training" in filepath else "testing"
        img_dir = os.path.join(img_dir, split+"_data")
        for i, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
                doc = data["document"]
            n_entity = len(data["label_entities"])
            n_relation = len(data["label_linkings"])

            image_path = os.path.join(img_dir, data["img"]["fname"])
            image, size = load_image(image_path)
            
            # iterate through data["document"] to strip off empty words,
            # and get a mapping from the origin id to the id after stripping 
            origin_id_to_stripped_id = {}
            stripped_id = 0
            for item in doc:
                cur_line_bboxes = []
                words= item["words"]
                stripped_words = []
                for w in words:
                    if w["text"].strip() == "":
                        continue
                    wid = w["id"]
                    origin_id_to_stripped_id[wid] = stripped_id
                    stripped_id += 1
                    stripped_words.append(w)
                    tokens.append(w["text"])
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
                if len(stripped_words) == 0:
                    continue
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # box = normalize_bbox(item["box"], size)
                # cur_line_bboxes = [box for _ in range(len(words))]
                bboxes.extend(cur_line_bboxes)
            
            # decide the start and end of every entity based on origin_id_to_stripped_id
            label_id_map = {"header": 0, "question": 1, "answer": 2, "other": 3}
            entities = []
            for i, entity in enumerate(data["label_entities"]):
                assert i == entity["entity_id"]
                cur_entity = {}
                n_entity_words = len(entity["word_idx"])
                assert n_entity_words > 0
                prev_word_idx = -1
                for j in range(n_entity_words):
                    if j == 0:
                        prev_word_idx = entity["word_idx"][j]
                        continue
                    assert entity["word_idx"][j] == prev_word_idx + 1
                    prev_word_idx = entity["word_idx"][j]
                for start_i in range(n_entity_words):
                    if entity["word_idx"][start_i] in origin_id_to_stripped_id:
                        cur_entity["start"] = origin_id_to_stripped_id[entity["word_idx"][start_i]]
                        break
                for end_i in range(n_entity_words-1, -1, -1):
                    if entity["word_idx"][end_i] in origin_id_to_stripped_id:
                        cur_entity["end"] = origin_id_to_stripped_id[entity["word_idx"][end_i]] + 1
                        break
                cur_entity["label"] = label_id_map[entity["label"]]
                assert cur_entity["label"] != 3
                assert len(cur_entity) == 3
                entities.append(cur_entity)    
            
            relations = []
            
            for relation in data["label_linkings"]:
                cur_relation = {}
                assert relation[0] < n_entity and relation[1] < n_entity
                cur_relation["head"], cur_relation["tail"] = relation
                relations.append(cur_relation) 

            uid = data["uid"]
            yield uid, {
                "id": uid,
                "tokens": tokens,
                "bboxes": bboxes,
                "entities": entities,
                "relations": relations,
                "image": image,
                "image_path": image_path,
            }