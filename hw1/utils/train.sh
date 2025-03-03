#!/bin/bash
python mc.py \
    --train_file   $1 \
    --validation_file $2 \
    --output_dir $3 \
    --per_device_train_batch_size  4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --weight_decay 0.1 \
    --model_name_or_path hfl/chinese-lert-base \
    --max_seq_length  512 \
    --learning_rate  5e-5

python3 qa.py \
    --train_file   $4 \
    --validation_file $5 \
    --output_dir $6 \
    --per_device_train_batch_size  4 \
    --gradient_accumulation_steps 4 \
    --num_train_epoch 5 \
    --weight_decay 0.1 \
    --model_name_or_path hfl/chinese-lert-large \
    --max_seq_len  512 \
    --learning_rate  5e-5 \