#!/usr/bin/env python
# coding=utf-8
import pyarrow
pyarrow.PyExtensionType.set_auto_load(True)
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset

import transformers

from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from layoutlmft.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3ForRelationExtraction
from layoutlmft.models.layoutlmv3.tokenization_layoutlmv3_fast import LayoutLMv3Tokenizer, LayoutLMv3TokenizerFast
from layoutlmft.trainer.funsd_trainer import FunsdReTrainer

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import torch

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    debug_mode: bool = field(default=False, metadata={"help": "debug mode"})
    task_name: Optional[str] = field(default="re", metadata={"help": "The name of the task (re, ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name == 'funsd':
        import layoutlmft.data.funsd_re
        datasets = load_dataset(os.path.abspath(layoutlmft.data.funsd_re.__file__), cache_dir=model_args.cache_dir)
    elif data_args.dataset_name == 'cord':
        import layoutlmft.data.cord
        datasets = load_dataset(os.path.abspath(layoutlmft.data.cord.__file__), cache_dir=model_args.cache_dir)
    elif data_args.dataset_name == 'custom':
        import layoutlmft.data.custom_re
        datasets = load_dataset(os.path.abspath(layoutlmft.data.custom_re.__file__), cache_dir=model_args.cache_dir)
    else:
        raise NotImplementedError()

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"

    remove_columns = column_names

    num_labels = 2

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = LayoutLMv3ForRelationExtraction.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])

        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    # Tokenize all texts and align the start/end indices to the tokenizer
    def tokenize_and_align_start_end(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        bboxes = []
        images = []
        entities = []
        batch_size = len(tokenized_inputs["input_ids"])
        for batch_index in range(batch_size):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            start_word_idxs = examples["entities"][batch_index]["start"]
            end_word_idxs = examples["entities"][batch_index]["end"]
            aligned_start_word_idxs = []
            aligned_end_word_idxs = []
            bbox = examples["bboxes"][batch_index]
            previous_word_idx = None
            # Outer loop: for every entity(bbox_inputs has the same outcome every loop, but to keep the code clean, we put it here)
            for start_word_idx, end_word_idx in zip(start_word_idxs, end_word_idxs):
                bbox_inputs = []
                start_token_idx = None
                end_token_idx = None
                # Inner loop: one pass word_ids to find the start/end token idx(end token idx is the first token of the next word)
                for token_idx, word_idx in enumerate(word_ids):
                    # Special tokens have a word id that is None.
                    if word_idx is None:
                        bbox_inputs.append([0, 0, 0, 0])
                    # the first token of each word.
                    elif word_idx != previous_word_idx:
                        bbox_inputs.append(bbox[word_idx])
                        if word_idx == start_word_idx:
                            start_token_idx = token_idx
                        if word_idx == end_word_idx:
                            end_token_idx = token_idx
                    else:
                        bbox_inputs.append(bbox[word_idx])
                    previous_word_idx = word_idx

                assert start_token_idx is not None
                if end_token_idx is None:
                    end_token_idx = len(word_ids) - 1 # -1 because the last token is a special token
                aligned_start_word_idxs.append(start_token_idx)
                aligned_end_word_idxs.append(end_token_idx)
                
            bboxes.append(bbox_inputs)
            entities.append({
                "start": aligned_start_word_idxs,
                "end": aligned_end_word_idxs,
                "label": examples["entities"][batch_index]["label"],
            })

            if data_args.visual_embed:
                ipath = examples["image_path"][batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["entities"] = entities
        tokenized_inputs["relations"] = examples["relations"]
        # dummy labels because the trainer needs them
        tokenized_inputs["labels"] = [10000] * batch_size
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_start_end,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        validation_name = "test"
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.debug_mode:
            eval_dataset = train_dataset
        else:
            eval_dataset = datasets[validation_name]
            if data_args.max_val_samples is not None:
                eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
            eval_dataset = eval_dataset.map(
                tokenize_and_align_start_end,
                batched=True,
                batch_size=64,
                remove_columns=remove_columns,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        if data_args.debug_mode:
            test_dataset = train_dataset
        else:
            test_dataset = datasets["test"]
            if data_args.max_test_samples is not None:
                test_dataset = test_dataset.select(range(data_args.max_test_samples))
            test_dataset = test_dataset.map(
                tokenize_and_align_start_end,
                batched=True,
                batch_size=64,
                remove_columns=remove_columns,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=1024,
    )

    def compute_metrics(p):
        batched_pred_relations = p.predictions
        batched_relations = p.label_ids
        tp = fp = fn = 0
        for pred_relations, relations in zip(batched_pred_relations, batched_relations):
            pred_relations = set([(r["head_id"], r["tail_id"]) for r in pred_relations])
            relations = set([(r["head_id"], r["tail_id"]) for r in relations])
            tp += len(pred_relations & relations)
            fp += len(pred_relations - relations)
            fn += len(relations - pred_relations)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
            

    # Initialize our Trainer
    trainer = FunsdReTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.argmax(predictions, axis=1)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
