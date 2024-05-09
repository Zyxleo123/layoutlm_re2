# 训练细节

## Relation Extraction(RE)

1. `layoutlmft/data/funsd_re` / `layoutlmft/data/custom_re`(数据集builder)

    Schema(除图像和bounding box部分)

    ```json
    "id": "..."
    "tokens": ["..."],
    "bboxes": [[...]]
    "entities": [
        {
            "start": [...]
            "end": [...]
            "label": [...]
        }
    ],
    "relations": [
        {
            "head": [...]
            "tail": [...]
        }
    ],
    "image": [[[...]]],
    "image_path": "..."
    ```

    解释：`entities`中的`start`和`end`是对应的`tokens`中的位置，界定实体的左闭右开区间端点，`label`是对应的实体类别，有0123四个值。`relations`中的`head`和`tail`是对应的`entities`中的位置。

    生成策略：只是详实地把原始数据的标注信息转换成了上述格式，没有做额外的处理，除了将`word["text"].strip()`后为空的词语从`tokens`中删除，如果一个entity都是空词，就略过这个entity(没有去除"\<unk\>"，实测ner上区别不大)。这使得`start`，`end`和`tokens`中的位置一一对应，而不一定和原始文档的标注word_id对应。

2. `examples/run_re.py:tokenize_and_align_start_end`(Dataset.map函数的函数参数；tokenization，以及将原来的`start`, `end`和tokenize之后的`input_ids`对齐)

    目的是tokenize`tokens`，并且对齐`relations["start]`，`relations["end"]`。细节略，映射过后可以直接输入模型。包含`input_ids`，`attention_mask`，`bbox`，`relations`，`entities`，`images`。我还加了dummy `labels`，值都是10000，它的存在只是让trainer不报错(也许可以去掉，但是我没试过)。

    另外，trainer被subclass了，改了evaluation/prediction的logic，目的仅仅是适应特别的输入输出（主要是因为有`relations`，`entities`输入；且没有有效的labels，要在下述的细节中生成）。代码在`layoutlmft/trainer/funsd_trainer.py`(名字有误导性，也可以train custom dataset)。新的trainer没有任何改变训练逻辑的地方。

3. `layoutlmft/models/re.py:build_relation`（model计算损失前，生成正负关系）

    下面的`entities`和`relations`就是上面得到的对象。源代码中，`entities`和`relations`都是batched的，先索引batch维。下面描述的是单个样本。

    先遍历`len(entities)*len(entities)`次得到所有可能的关系`all_possible_relations`。注意合法的关系当且仅当(e_head, e_tail)中，e_head的label属于question，e_tail的label为answer，比如e_head==e_tail不可能是合法的关系。

    然后，`relations`也按照这个要求，去掉所有不合法的关系。剩余的关系是正例`positive_relations`。

    从`all_possible_relations`做和`positive_relations`的差集，得到负例`negative_relations`。

    将`positive_relations`和`negative_relations` concat，并且为前者加上标签1，后者加上标签0，为每个batch处理后返回。此时返回的`relations`是一个列表，每个元素是一个字典，包含`head`，`tail`和`label`三个key（前两者的含义与dataset schema描述的相同）。

    例：`entities`中有8 questions, 3 answers；`relations`中有4个(question, answer)对。那么`all_possible_relations`有24个，`positive_relations`有4个，`negative_relations`有20个。返回的`relations`是一个长度为24的列表，每个元素是一个字典，包含`head`，`tail`和`label`三个key。

4. `examples/run_re.py:compute_metrics`（计算f1等）

    函数输入2个形如以下列表的两个参数，其中一个代表模型**预测为真**的所有关系，另一个是真实值:

    ```python
    [
        [{'head_id': 4, ...}, ...],
        [{'head_id': 40, ...}, ...],
    ]
    ```

    第一维是batch，索引batch以后得到关系列表，每个元素是一个字典，包含`head_id`，`tail_id`，其余key不重要。

    接下来遍历所有batch，通过gt和pred两个集合的交差并运算，统计每个batch的TP, FP, FN，累加得到总的TP, FP, FN。最后计算f1等值。

## Named Entity Recognition(NER)

1. (没有更改)`layoutlmft/data/funsd_ner` / (新加的)`layoutlmft/data/custom_ner`(数据集builder)

    Schema(除图像和bounding box部分)

    ```json
    "id": "...",
    "tokens": ["..."],
    "bboxes": [[...]],
    "ner_tags": ["O", "B-<tag1>", "I-<tag1>", "B-<tag2>", "I-<tag2>", ...],
    "image": [[[...]]],
    "image_path": "..."
    ```

    同样只是详实地把原始数据的标注信息转换成了上述格式，没有做额外的处理，以及去除空词。标注`ner_tags`的细节略。

2. (没有更改)`examples/run_ner.py:tokenize_and_align_labels`(Dataset.map函数的函数参数；tokenization，以及将原来的`ner_tags`和tokenize之后的`input_ids`对齐)

    目的是为tokenized后的subwords标注`ner_tags`，可以选择`label_all_tokens`，也就是subwords共享一个label；或者不选这个选项，那么就只有一个subword有label，剩余的都是-100，loss和metric计算都会忽略。

3. (没有更改)`examples/run_ner.py:compute_metrics`（计算f1等）

    去除label=-100的prediction和label后，调用`classification_report`输出。
    
# 3 Random test results

seed is fed to `random`, `numpy`, `torch`, `torch.cuda`

| seed/task&dataset | funsd_re | custom_re | funsd_ner | custom_ner |
| :---: | :---: | :---: | :---: | :---: |
| 21 | 0.6972 | 0.6798 | 0.9074 | 0.8246 |
| 28 | 0.6839 | 0.6673 | 0.9073 | 0.8196 |
| default | 0.7131 | 0.6772 | 0.911 | 0.8248 |

# 4 RO results

seed is fed to `random`, `numpy`, `torch`, `torch.cuda`

| seed/task&dataset | ori_re | ie_re | ori_ner | ie_ner |
| :---: | :---: | :---: | :---: | :---: |
| 21 | 0.6927 | 0.6958 | 0.9049 | 0.8226 |
| 28 | 0.682 | 0.6762 | 0.9067 | 0.8237 |
| default | 0.729 | 0.7292 | 0.9165 | 0.8359 |

Hyperparameters:

| dataset_task | initial lam | ro layers | lam lr
| :---: | :---: | :---: | :---: |
| ner_ie | 0.1 | 12 | 0.02 | 
| ner_ori | 1.0 | 7 | 0.1 |
| kre_ie | 10 | 12 | 0 |
| re_ori | 5 | 12 | 0.01 |

# 5 SROIE & CORD NER results(RO & without RO)

All trained on 1000 Steps, total batch size is 16(including gradient accumulation) for sroie, and 64 for cord.

| Use RO\Dataset-Model scale | SROIE-base | SROIE-large | CORD-base | CORD-large |
| :---: | :---: | :---: |
| True | 0.9565 | 0.9659 | 0.9672 | 0.9729 |
| False | 0.948 | 0.9612 | 0.9591 | 0.9698 |

Hyperparameters:

| Dataset-Model scale | initial lam | ro layers(not searched) | lam lr | lr(searched preliminarily, not with ro params) |
| :---: | :---: | :---: | :---: | :---: |
| SROIE-base | 1.0 | 12 | 0.01 | 7e-5 |
| SROIE-large | 10.0 | 24 | 0.1 | 3e-5 |
| CORD-base | 0.1 | 12 | 0.5 | 1e-4 |
| CORD-large | 0.01 | 24 | 0.05 | 5e-5 |

