export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"

# seed=21

# export CUDA_VISIBLE_DEVICES=2

# nohup python examples/run_re.py \
#   --dataset_name funsd \
#   --do_train \
#   --do_eval \
#   --model_name_or_path ./layoutlmv3-base-1028 \
#   --segment_level_layout 1 \
#   --visual_embed 1 \
#   --input_size 224 \
#   --learning_rate 1e-5 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --dataloader_num_workers 8 \
#   \
#   --output_dir ./results/layoutlmv3-base-finetuned-funsd-re-new \
#   --overwrite_output_dir yes \
#   \
#   --num_train_epochs 400 \
#   --evaluation_strategy steps \
#   --eval_steps 1000 \
#   --save_steps -1 \
#   --seed 28 \
#   \
#   --logging_steps 1000 \
#   --logging_dir ./logs-funsd-re/seed-28 > ./logs-funsd-re/seed-28/nohup.out 2>&1 &

export CUDA_VISIBLE_DEVICES=4

nohup python examples/run_ner.py \
  --dataset_name funsd \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-base-1028 \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --dataloader_num_workers 8 \
  --output_dir ./results/layoutlmv3-base-finetuned-funsd-ner \
  --overwrite_output_dir yes \
  --max_steps 1000 \
  --save_steps -1 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --learning_rate 3e-5 \
  --logging_steps 1 \
  --logging_dir ./logs-funsd-ner/seed-21 \
  --seed 21 > ./logs-funsd-ner/seed-21/nohup.out 2>&1 &

export CUDA_VISIBLE_DEVICES=6

nohup python examples/run_ner.py \
  --dataset_name funsd \
  --do_train \
  --do_eval \
  --model_name_or_path ./layoutlmv3-base-1028 \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --dataloader_num_workers 8 \
  --output_dir ./results/layoutlmv3-base-finetuned-funsd-ner \
  --overwrite_output_dir yes \
  --max_steps 1000 \
  --save_steps -1 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --learning_rate 3e-5 \
  --logging_steps 1 \
  --logging_dir ./logs-funsd-ner/seed-28 \
  --seed 28 > ./logs-funsd-ner/seed-28/nohup.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=4

# nohup python examples/run_re.py \
#   --dataset_name custom \
#   --do_train \
#   --do_eval \
#   --model_name_or_path ./layoutlmv3-base-1028 \
#   --output_dir ./results/layoutlmv3-base-finetuned-custom-re \
#   --overwrite_output_dir yes \
#   --segment_level_layout 1 \
#   --visual_embed 1 \
#   --input_size 224 \
#   --save_steps -1 \
#   --learning_rate 1e-5 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 1 \
#   --dataloader_num_workers 8 \
#   --num_train_epochs 400 \
#   --evaluation_strategy steps \
#   --eval_steps 1000 \
#   --save_steps 1000 \
#   --logging_steps 1000 \
#   --logging_dir ./logs-custom-re/seed-28 \
#   --seed 28 > ./logs-custom-re/seed-28/nohup.out 2>&1 &

# export CUDA_VISIBLE_DEVICES=6

# nohup python examples/run_ner.py \
#   --dataset_name custom \
#   --do_train \
#   --do_eval \
#   --model_name_or_path ./layoutlmv3-base-1028 \
#   --output_dir ./results/layoutlmv3-base-finetuned-custom-ner \
#   --overwrite_output_dir yes \
#   --segment_level_layout 1 \
#   --visual_embed 1 \
#   --input_size 224 \
#   --max_steps 1000 \
#   --save_steps -1 \
#   --evaluation_strategy steps \
#   --eval_steps 100 \
#   --per_device_train_batch_size 1 \
#   --gradient_accumulation_steps 16 \
#   --dataloader_num_workers 8 \
#   --logging_steps 1 \
#   --logging_dir ./logs-custom-ner/seed-28 \
#   --learning_rate 7e-5 \
#   --seed 28 > ./logs-custom-ner/seed-28/nohup.out 2>&1 &
