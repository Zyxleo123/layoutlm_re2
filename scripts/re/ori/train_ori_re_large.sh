# conda init 
# conda activate layoutlmv3
# 目前只支持单卡

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=6
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"

python examples/run_re.py \
  --dataset_name custom-ori \
  --ro_info \
  --lam 10 \
  --ro_layers 12 \
  --lam_lr 1e-5 \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-large-re-1028-ft \
  --output_dir ./results/layoutlmv3-base-finetuned-custom-re \
  --overwrite_output_dir yes \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --save_steps -1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --dataloader_num_workers 8 \
  --num_train_epochs 400 \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --logging_steps 1 \
  --logging_dir ./logs-large/re/ori/run/10-12-1e-05-42 \
  --seed 42 \
