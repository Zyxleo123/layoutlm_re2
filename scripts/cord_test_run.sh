# conda init 
# conda activate layoutlmv3
# 目前只支持单卡

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layoutlm_re2:$PYTHONPATH"

python examples/run_ner.py \
  --dataset_name cord \
  --ro_info \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-large-514 \
  --output_dir ./results/test1 \
  --overwrite_output_dir yes \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --max_steps 10 \
  --save_steps -1 \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --dataloader_num_workers 8 \
  --logging_steps 1 \
  --logging_dir ./logs/test1 \
  --learning_rate 3e-5 \
  --lam 15.0 \
  --ro_layers 3 \
