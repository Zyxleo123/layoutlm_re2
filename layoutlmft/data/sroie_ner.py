# coding=utf-8
'''
Reference: https://huggingface.co/datasets/nielsr/funsd/blob/main/funsd.py
'''
import json
import os
import re

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

    def split_with_indices(input_str, base_index):
        # Note: must match leading spaces instead of trailing spaces.
        #       Because Date entities in texts like ": 2013/1/1 Open" is always annotated with the leading space, excluding the trailing space.
        pat = re.compile(r"\s*\S+")
        matches = pat.finditer(input_str)
        result_text = []
        result_range = []
        for match in matches:
            result_text.append(match.group())
            result_range.append(list(range(match.start()+base_index, match.end()+base_index)))
        # assert that result_range is a partition of range(len(input_str))
        assert len(set(sum(result_range, []))) == len(input_str)
        for i in range(len(result_range)):
            for j in range(i+1, len(result_range)):
                assert len(set(result_range[i]) & set(result_range[j])) == 0

        return result_text, result_range

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    class Token:
        def __init__(self, text: str, word_ids: list):
            self.text = text
            self.word_ids = word_ids
            self.label = None

    def four_point_to_box(points):
        assert len(points) == 4
        return points[0] + points[2]

    def correct_bbox(box, size):
        if any([box[i] < 0 or box[i] > size[i % 2] for i in range(4)]):
            box = [0 if box[i] < 0 else box[i] for i in range(4)]
            print(box)
        if any([box[i] > size[i % 2] for i in range(4)]):
            box = [size[i % 2] if box[i] > size[i % 2] else box[i] for i in range(4)]
            print(box)
        return box

    def four_point_to_box(self, points):
        assert len(points) == 4
        return points[0] + points[2]

    def _generate_examples(self, split):
        anno_full = os.path.join(self.data_root, self.anno_dir)
        img_full = os.path.join(self.data_root, self.img_dir)

        anno_files = os.listdir(anno_full)
        for uid, file in enumerate(anno_files): 
            with open(os.path.join(anno_full, file), 'r') as f:
                data = json.load(f)
                image_file = data['uid'] + '.jpg'
                assert image_file == file.replace('.json', '.jpg')
                image_path = os.path.join(img_full, image_file)
                annotated_size = data['img']['width'], data['img']['height']
                image = load_image(image_path)
                document = data['document']
                tokens = []
                bboxes = []
                word_id2token_id = {}
                total_word_count = 0
                # whitespace tokenize and do two-way mapping between word_id(annotated id) and token_id(output list index)
                for segment in document:
                    seg_text = segment['text']
                    texts, word_ids = self.split_with_indices(seg_text, total_word_count)
                    assert len(texts) == len(word_ids)

                    box = segment['box']
                    box = self.four_point_to_box(box)
                    box = self.correct_bbox(box, annotated_size)
                    box = normalize_bbox(box, annotated_size)
                    bboxes.extend([box] * len(texts))

                    for i, word_id in enumerate(word_ids):
                        for id in word_id:
                            word_id2token_id[id] = len(tokens) + i
                    total_word_count += len(segment['words'])
                    tokens.extend([self.Token(text, word_ids) for text, word_ids in zip(texts, word_ids)])

                # assign labels to tokens
                assigned = [False] * len(data['label_entities'])
                for token in tokens:
                    text = token.text
                    # strip leading/trailing whitespaces
                    start = 0
                    end = len(text)
                    ## Note: following code is used to ignore leading/trailing whitespaces while assigning a label. 
                    ##       It turns out not to be a concern in SROIE dataset.
                    # while start < len(text) and text[start].isspace():
                    #     start += 1
                    # while end > 0 and text[end-1].isspace():
                    #     end -= 1
                    # # assert that at most 1 space removed in both ends
                    # assert start <= 1 and len(text) - end <= 1, (token.text, anno_file)
                    word_ids = set(token.word_ids[start:end])

                    for entity in data['label_entities']:
                        label = entity['label'] + '+' + str(entity['entity_id'])
                        assert len(entity['word_idx']) <= 1
                        if len(entity['word_idx']) == 0:
                            assigned[entity['entity_id']] = True
                            continue
                        entity_word_ids = set(entity['word_idx'][0])
                        if (word_ids & entity_word_ids) and not word_ids.issubset(entity_word_ids):
                            print(token.text, file, sorted(list(word_ids)), sorted(list(entity_word_ids)))

                        if word_ids.issubset(entity_word_ids):
                            token.label = label
                            assigned[entity['entity_id']] = True
                            break
                        
                assert all(assigned), (file, assigned)
                data_entity_word_ids = []
                gt_entity_word_ids = sum([entity['word_idx'][0] for entity in data['label_entities'] if len(entity['word_idx']) > 0], [])
                for token in tokens:
                    if token.label is not None:
                        data_entity_word_ids.extend(token.word_ids)
                assert set(data_entity_word_ids) == set(gt_entity_word_ids), (file, data_entity_word_ids, gt_entity_word_ids)


                # add B- and I- prefix to labels; iterate reversely to avoid changing labels of previous tokens
                for i in range(len(tokens)-1, -1, -1):
                    if tokens[i].label is None:
                        tokens[i].label = 'O'
                    elif i == 0 or tokens[i-1].label != tokens[i].label:
                        # remove '+{id}' suffix
                        label = tokens[i].label.split('+')[0]
                        tokens[i].label = 'B-' + label
                    else:
                        label = tokens[i].label.split('+')[0]
                        tokens[i].label = 'I-' + label

                ro_spans = []
                if self.ro_info:
                    for linking in data["ro_linkings"]:
                        cur_span = {}
                        head_seg, tail_seg = document[linking[0]], document[linking[1]]
                        cur_span['head_start'] = word_id2token_id[head_seg['words'][0]['id']]
                        cur_span['head_end'] = word_id2token_id[head_seg['words'][-1]['id']] + 1
                        cur_span['tail_start'] = word_id2token_id[tail_seg['words'][0]['id']]
                        cur_span['tail_end'] = word_id2token_id[tail_seg['words'][-1]['id']] + 1
                        ro_spans.append(cur_span)

                # add to output
                yield uid, {'id': uid, 'tokens': [token.text for token in tokens], 'word_ids': [token.word_ids for token in tokens], 
                        'ner_tags': [token.label for token in tokens], 'bboxes': bboxes, 'image_path': image_path, 'image': image,
                        'ro_spans': ro_spans}