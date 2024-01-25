#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"
learning_rates=(1e-5 3e-5 5e-5 7e-5 9e-5 1e-6 3e-6 5e-6 7e-6 9e-6 1e-4 3e-4 5e-4 7e-4 9e-4 1e-3)

for learning_rate in "${learning_rates[@]}"; do
python examples/run_ner.py \
  --dataset_name funsd \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-base-1028 \
  --output_dir ./results/layoutlmv3-base-finetuned-funsd-ner \
  --overwrite_output_dir yes \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --max_steps 1000 \
  --save_steps -1 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --dataloader_num_workers 8 \
  --learning_rate $learning_rate > ./logs-funsd-ner/lr-$learning_rate.log 2>&1
done
