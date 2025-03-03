#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 && python3 summarization.py \
    --train_file   $1 \
    --validation_file $2 \
    --text_column maintext \
    --summary_column title \
    --output_dir $3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 20 \
    --model_name_or_path google/mt5-small \
    --learning_rate  3e-4 \
    --max_target_length 1024 \
    --max_source_length 2048 \
    --source_prefix "summarize: " \



