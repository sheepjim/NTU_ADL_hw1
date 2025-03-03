# ADL/hw1

## Download
`bash ./download.sh`
-modelMC
-modelQA

## Process the input
`python ./huggingface_input_processor.py`
- Data
    -trainMC.json
    -trainQA.json
    -validMC.json
    -validQA.json
    -test.json

## Train
`bash ./train.sh ${path_to_trainMC.json} ${path_to_validMC.json} ${path_to_outputMC_dir} ${path_to_trainQA.json} ${path_to_validQA.json} ${path_to_outputQA_dir}`

## Infer
`bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv`
