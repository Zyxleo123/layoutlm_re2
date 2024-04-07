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

Label2ids = {}
with open('./RO-Datasets/SROIE/labels_bio.txt', 'r') as f:
    idx = 0
    for line in f:
        label = line.strip()
        Label2ids[label] = idx
        idx += 1

Train_files = []
with open('./RO-Datasets/SROIE/data.train.txt', 'r') as f:
    for line in f:
        Train_files.append(line.split('\t')[-1].split('/')[-1].strip())

Test_files = []
with open('./RO-Datasets/SROIE/data.test.txt', 'r') as f:
    for line in f:
        Test_files.append(line.split('\t')[-1].split('/')[-1].strip())
    
class CustomNERConfig(datasets.BuilderConfig):
    """BuilderConfig for out custom dataset"""

    def __init__(self, data_root, anno_dir, img_dir,
                 ro_info,
                 train_files, test_files, **kwargs): 
        """BuilderConfig for our custom dataset.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CustomNERConfig, self).__init__(**kwargs)
        self.data_root = data_root
        self.anno_dir = anno_dir
        self.img_dir = img_dir
        self.ro_info = ro_info
        self.train_files = train_files
        self.test_files = test_files


class Sroie(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIGS = [
        CustomNERConfig(name="default", version=datasets.Version("1.0.0"), description="Sroie", 
                        data_root=os.path.join("RO-Datasets", "SROIE"), anno_dir="jsons", img_dir="images",
                        ro_info=True,
                        train_files=Train_files, test_files=Test_files)
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
                            names=list([key.upper() for key in Label2ids.keys()])
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
        sroie_root = os.path.join('RO-Datasets', 'SROIE')
        anno_dir = self.config.data_dir
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"split": "train"}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"split": "test"}
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def four_point_to_box(self, points):
        assert len(points) == 4
        return points[0] + points[2]

    def _generate_examples(self, split):
        anno_full = os.path.join(self.config.data_root, self.config.anno_dir)
        img_full = os.path.join(self.config.data_root, self.config.img_dir)

        logger.info("⏳ Generating examples from = %s", anno_full)
        for uid, file in enumerate(sorted(os.listdir(anno_full))):
            if split == "train" and file not in self.config.train_files or \
               split == "test" and file not in self.config.test_files:
                continue
            tokens = []
            bboxes = []
            ner_tags = []

            file_path = os.path.join(anno_full, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_file = file.replace('.json', '.jpg')
            assert image_file == data['uid'] + '.jpg'
            image_path = os.path.join(img_full, image_file)
            image, size = load_image(image_path)
            annotated_size = data['img']['width'], data['img']['height']
            assert size == annotated_size

            # label each original word
            max_word_id = data["document"][-1]["words"][-1]["id"]
            word_id_to_entity_label = [-1] * (max_word_id + 1)
            for i, entity in enumerate(data["label_entities"]):
                entity_label = entity["label"].upper()
                assert len(entity["word_idx"]) <= 1, file+' '+str(i)
                if len(entity["word_idx"]) == 0:
                    continue
                for word_id in entity["word_idx"][0]:
                    word_id_to_entity_label[word_id] = entity_label + '-' + str(entity["entity_id"])
            for word_id in range(len(word_id_to_entity_label)):
                if word_id_to_entity_label[word_id] == -1:
                    word_id_to_entity_label[word_id] = "O"
            assert np.array(word_id_to_entity_label == -1).sum() == 0

            # remove empty words; map annotation word id to token index in 'tokens'
            # tokens_word_id map each token to its annotation word id
            origin_id_to_stripped_id = {}
            stripped_id = 0
            tokens_word_id = []
            for item in data["document"]:
                cur_line_bboxes = []
                words = item["words"]
                stripped_words = []
                for w in words:
                    if w["text"] == "":
                        continue
                    wid = w["id"]
                    origin_id_to_stripped_id[wid] = stripped_id
                    stripped_id += 1
                    stripped_words.append(w)
                    tokens.append(w["text"])
                    tokens_word_id.append(wid)

                    box = self.four_point_to_box(w["box"])
                    if any([box[i] < 0 or box[i] > size[i % 2] for i in range(4)]):
                        print("File {} bbox is not proper: {}; Segment: {}".format(file, box, item))
                        box = [0 if box[i] < 0 else box[i] for i in range(4)]
                        print("After correction: {}".format(box))
                    elif any([box[i] > size[i % 2] for i in range(4)]):
                        print("File {} bbox is not proper: {}; Segment: {}".format(file, box, item))
                        box = [size[i % 2] if box[i] > size[i % 2] else box[i] for i in range(4)]
                        print("After correction: {}".format(box))
                    normalized_bbox = normalize_bbox(box, size)
                    cur_line_bboxes.append(normalized_bbox)
                    normalized_bbox = normalize_bbox(box, size)
                    cur_line_bboxes.append(normalized_bbox)

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
                    # if word_id == 0 or word_id_to_entity_label[word_id - 1] != word_id_to_entity_label[word_id]:
                    if len(ner_tags) == 0 or word_id_to_entity_label[word_id] != word_id_to_entity_label[word_id - 1]:
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
            
            yield uid, {"id": uid, "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags, 
                         "ro_spans": ro_spans, "image": image, "image_path": image_path}