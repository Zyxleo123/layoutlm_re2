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


class FunsdReConfig(datasets.BuilderConfig):
    """BuilderConfig for FUNSD"""

    def __init__(self, **kwargs):
        """BuilderConfig for FUNSD.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(FunsdReConfig, self).__init__(**kwargs)


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        FunsdReConfig(name="funsd", version=datasets.Version("1.0.0"), description="FUNSD dataset"),
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
        downloaded_file = dl_manager.download_and_extract("https://guillaumejaume.github.io/FUNSD/dataset.zip")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"filepath": f"{downloaded_file}/dataset/training_data/"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"filepath": f"{downloaded_file}/dataset/testing_data/"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")

        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            tokens = []
            bboxes = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
                form = data["form"]

            n_entities = len(form)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in form:
                cur_line_bboxes = []
                words= item["words"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                for w in words:
                    tokens.append(w["text"])
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                # box = normalize_bbox(item["box"], size)
                # cur_line_bboxes = [box for _ in range(len(words))]
                bboxes.extend(cur_line_bboxes)
            
            label_id_map = {"header": 0, "question": 1, "answer": 2, "other": 3}
            entities = []
            entity_start = 0
            entity_stripped_mapping = {}
            new_entity_id = 0
            for entity_id in range(n_entities):
                cur_entity = {}
                assert entity_id == form[entity_id]["id"]
                e_words= form[entity_id]["words"]
                e_words = [w for w in e_words if w["text"].strip() != ""]
                if len(e_words) == 0:
                    continue
                cur_entity["start"] = entity_start
                entity_start += len(e_words)
                cur_entity["end"] = entity_start
                cur_entity["label"] = label_id_map[form[entity_id]["label"]]
                entities.append(cur_entity)
                entity_stripped_mapping[entity_id] = new_entity_id
                new_entity_id += 1
            
            relations = set()
            for entity_id in range(n_entities):
                cur_relations = set()
                for relation in form[entity_id]["linking"]:
                    if relation[0] not in entity_stripped_mapping or relation[1] not in entity_stripped_mapping:
                        continue
                    cur_relations.add((entity_stripped_mapping[relation[0]], entity_stripped_mapping[relation[1]]))
                relations.update(cur_relations)
            relations = list(relations)
            relations = [{"head": r[0], "tail": r[1]} for r in relations]
            relations = sorted(relations, key=lambda x: x["head"])
            try:
                assert len(bboxes) == len(tokens)
            except AssertionError:
                print(f"Assertion failed. Length of bboxes: {len(bboxes)}, tokens: {len(tokens)}, entities: {len(entities)}")
                raise 
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "bboxes": bboxes,
                "entities": entities,
                "relations": relations,
                "image": image,
                "image_path": image_path,
            }
            
