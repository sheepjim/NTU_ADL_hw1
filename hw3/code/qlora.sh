export CUDA_VISIBLE_DEVICES=1 && python qlora.py \
  --model_name_or_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
  --output_dir $2 \
  --dataset $1 \
  --dataset_format adl-hw3 \
  --do_train True \
  --do_eval True\
  --max_steps 1000 \
  --save_steps 10 \
  --save_total_limit 100 \
  --learning_rate 2e-4 \
  --lr_scheduler_type inverse_sqrt \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_r 64 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \


