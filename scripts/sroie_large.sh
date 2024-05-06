# conda init 
# conda activate layoutlmv3
# 目前只支持单卡

export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"

python examples/run_sroie_ner.py \
  --ro_info \
  --do_eval \
  --do_train \
  --model_name_or_path ./layoutlmv3-large-1028 \
  --output_dir ./results/layoutlmv3-base-ft-sroie-ner \
  --overwrite_output_dir yes \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --max_steps 1000 \
  --save_steps -1 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --dataloader_num_workers 8 \
  --logging_steps 1 \
  --logging_dir ./logs-sroie/test \
  --learning_rate 7e-5 \
  --lam 15.0 \
  --ro_layers 3 \