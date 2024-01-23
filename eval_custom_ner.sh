# conda init 
# conda activate layoutlmv3
# 目前只支持单卡

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"

python examples/run_ner.py \
  --dataset_name custom \
  --do_eval \
  --model_name_or_path ./layoutlmv3-base-finetuned-custom-ner \
  --output_dir ./results/layoutlmv3-base-finetuned-custom-ner-eval \
  --overwrite_output_dir yes \
  --segment_level_layout 1 \
  --visual_embed 1 \
  --input_size 224 \
  --dataloader_num_workers 1 \
