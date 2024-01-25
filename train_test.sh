# conda init 
# conda activate layoutlmv3
# 目前只支持单卡

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"

python examples/run_re.py \
  --dataset_name funsd \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-base-1028 \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 8 \
  \
  --output_dir ./results/test \
  --overwrite_output_dir yes \
  \
  --num_train_epochs 2 \
  --evaluation_strategy steps \
  --eval_steps 1 \
  --save_steps 1 \
  --load_best_model_at_end yes \
  \
  --logging_steps 1 \
  --logging_dir ./logs-new/test \
  \
  --max_train_samples 2 \
  --max_val_samples 1 \
