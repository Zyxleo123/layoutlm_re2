export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/root/layout:$PYTHONPATH"
cd /root/layout

ro_layers=(12 0 11 1 10 2 9 3 8 4 7 5 6)
lam=(1.0 5.0 0.5 2.0)

ro_layer=12

for lam in ${lam[@]}; do
    python examples/run_ner.py \
      --dataset_name custom-ie \
      --ro_info \
      --ro_layers $ro_layer \
      --lam $lam \
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
      --logging_dir "./logs/ner/ie/search/${ro_layer}_${lam}" \
      --learning_rate 7e-5 \
      --seed 1
done
