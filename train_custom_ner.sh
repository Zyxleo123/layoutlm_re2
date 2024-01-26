# conda init 
# conda activate layoutlmv3
# 目前只支持单卡

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"

python examples/run_ner.py \
  --dataset_name custom-default \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-base-1028 \
  --output_dir ./results/layoutlmv3-base-finetuned-custom-ner \
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
  --logging_steps 1 \
  --logging_dir ./logs-custom-ner \
  --learning_rate 7e-5 \
  --seed 21 \