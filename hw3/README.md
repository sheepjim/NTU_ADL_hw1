# Train
bash qlora.sh {path_to _train_data} {output_dir}

# Infer
bash download.sh
bash ./run.sh /path/to/model-folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json
