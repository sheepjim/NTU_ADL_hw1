#!/bin/bash 
export CUDA_VISIBLE_DEVICES=6 && python3 code/infer.py \
    --validation_file $1 \
    --output_file $2 \
    --text_column maintext \
    --num_beams 5 \
    --per_device_eval_batch_size 16 \
    --model_name_or_path model \
    --max_target_length 1024 \
    --max_source_length 2048 \
    --source_prefix "summarize: " \



