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


class CustomNERConfig(datasets.BuilderConfig):
    """BuilderConfig for out custom dataset"""

    def __init__(self, ro_info, **kwargs):
        """BuilderConfig for our custom dataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CustomNERConfig, self).__init__(**kwargs)
        self.ro_info = ro_info


class Funsd(datasets.GeneratorBasedBuilder):
    """Conll2003 dataset."""

    BUILDER_CONFIGS = [
        CustomNERConfig(name="default", version=datasets.Version("1.0.0"), description="Custom dataset(default)", data_dir="./layoutlmft/data/datasets/default", ro_info=False),
        CustomNERConfig(name="ie", version=datasets.Version("1.0.0"), description="Custom dataset(ie)", data_dir="./layoutlmft/data/datasets/ie", ro_info=True),
        CustomNERConfig(name="ori", version=datasets.Version("1.0.0"), description="Custom dataset(ori)", data_dir="./layoutlmft/data/datasets/ori", ro_info=True),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "bboxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION", "B-ANSWER", "I-ANSWER"]
                        )
                    ),
                    "ro_spans": datasets.Sequence(
                        {
                            "head_start": datasets.Value("int64"),
                            "head_end": datasets.Value("int64"),
                            "tail_start": datasets.Value("int64"),
                            "tail_end": datasets.Value("int64"),
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
        custom_ds = self.config.data_dir
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
        split = "training" if "train" in filepath else "testing"
        img_dir = os.path.join(img_dir, split+"_data")
        for file in sorted(os.listdir(ann_dir)):
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, data["img"]["fname"])
            image, size = load_image(image_path)

            max_word_id = data["document"][-1]["words"][-1]["id"]
            word_id_to_entity_label = [-1] * (max_word_id + 1)
            for entity in data["label_entities"]:
                entity_label = entity["label"].upper()
                for word_id in entity["word_idx"]:
                    word_id_to_entity_label[word_id] = entity_label + '-' + str(entity["entity_id"])
            for word_id in range(len(word_id_to_entity_label)):
                if word_id_to_entity_label[word_id] == -1:
                    word_id_to_entity_label[word_id] = "O"
            assert np.array(word_id_to_entity_label == -1).sum() == 0

            origin_id_to_stripped_id = {}
            stripped_id = 0
            tokens_word_id = []
            for item in data["document"]:
                cur_line_bboxes = []
                words = item["words"]
                stripped_words = []
                for w in words:
                    if w["text"].strip() == "":
                        continue
                    wid = w["id"]
                    origin_id_to_stripped_id[wid] = stripped_id
                    stripped_id += 1
                    stripped_words.append(w)
                    tokens.append(w["text"])
                    tokens_word_id.append(wid)
                    cur_line_bboxes.append(normalize_bbox(w["box"], size))
                if len(stripped_words) == 0:
                    continue
                # by default: --segment_level_layout 1
                # if do not want to use segment_level_layout, comment the following line
                cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                bboxes.extend(cur_line_bboxes)
            
            for word_id in tokens_word_id:
                if word_id_to_entity_label[word_id] == "O":
                    ner_tags.append("O")
                else:
                    if word_id == 0 or word_id_to_entity_label[word_id - 1] != word_id_to_entity_label[word_id]:
                        ner_tag = "B-" + word_id_to_entity_label[word_id]
                        ner_tag = ner_tag[:ner_tag.rfind('-')]
                        ner_tags.append(ner_tag)
                    else:
                        ner_tag = "I-" + word_id_to_entity_label[word_id]
                        ner_tag = ner_tag[:ner_tag.rfind('-')]
                        ner_tags.append(ner_tag)
            
            # read order
            ro_spans = []
            if self.config.ro_info:
                for linking in data["ro_linkings"]:
                    cur_span = {}
                    head_seg, tail_seg = linking
                    for word in data["document"][head_seg]["words"]:
                        if word["id"] in origin_id_to_stripped_id:
                            cur_span["head_start"] = origin_id_to_stripped_id[word["id"]]
                            break
                    for word in data["document"][tail_seg]["words"]:
                        if word["id"] in origin_id_to_stripped_id:
                            cur_span["tail_start"] = origin_id_to_stripped_id[word["id"]]
                            break
                    for word in data["document"][head_seg]["words"][::-1]:
                        if word["id"] in origin_id_to_stripped_id:
                            cur_span["head_end"] = origin_id_to_stripped_id[word["id"]] + 1
                            break
                    for word in data["document"][tail_seg]["words"][::-1]:
                        if word["id"] in origin_id_to_stripped_id:
                            cur_span["tail_end"] = origin_id_to_stripped_id[word["id"]] + 1
                            break
                    assert len(cur_span) == 4
                    ro_spans.append(cur_span)
            
            uid = data["uid"]
            yield uid, {"id": uid, "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, 
                         "ro_spans": ro_spans, "image": image, "image_path": image_path}