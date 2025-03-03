export CUDA_VISIBLE_DEVICES=5 && python shot.py \
    --base_model_path zake7749/gemma-2-2b-it-chinese-kyara-dpo \
    --peft_path models \
    --test_data_path data/shot.json \
    --output_path shot.json