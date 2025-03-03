import json
import numpy as np

with open('Data/context.json', 'r', encoding='utf-8') as f:
    context = json.load(f)
with open('Data/train.json', 'r', encoding='utf-8') as f:
    train = json.load(f)
with open('Data/valid.json', 'r', encoding='utf-8') as f:
    valid = json.load(f)
with open('Data/test.json', 'r', encoding='utf-8') as f:
    test = json.load(f)

trainMC = []
for item in train:
    
    Id = item["id"]
    sent1 = item["question"]
    
    label = -1    
    ending = []
    for i in range(0, 4):
        ending.append(context[item['paragraphs'][i]])
        if(item["relevant"] == item['paragraphs'][i]):
            label = i

    trainMC.append({
        "id": Id,
        "sent1": sent1,
        "sent2": "",
        "label": label,
        "ending0": ending[0],
        "ending1": ending[1],
        "ending2": ending[2],
        "ending3": ending[3],
    })


validMC = []
for item in valid:
    
    Id = item["id"]
    sent1 = item["question"]
    
    label = -1    
    ending = []
    for i in range(0, 4):
        ending.append(context[item['paragraphs'][i]])
        if(item["relevant"] == item['paragraphs'][i]):
            label = i

    validMC.append({
        "id": Id,
        "sent1": sent1,
        "sent2": "",
        "label": label,
        "ending0": ending[0],
        "ending1": ending[1],
        "ending2": ending[2],
        "ending3": ending[3],
    })


with open('Data/trainMC.json', 'w', encoding='utf-8') as f:
    json.dump(trainMC, f, ensure_ascii=False, indent=4)
with open('Data/validMC.json', 'w', encoding='utf-8') as f:
    json.dump(validMC, f, ensure_ascii=False, indent=4)


trainQA = []
for item in train:
    
    Id = item["id"]
    question = item["question"]
    Context = context[item['relevant']]
    answer = {"answer_start" : [int(np.int32(item['answer']['start']))], "text" : [item['answer']['text']]} 
    
    trainQA.append({
        "answers": answer,
        "context": Context,
        "id": Id,
        "question": question,
        "title": "train"
    })

validQA = []
for item in valid:
    
    Id = item["id"]
    question = item["question"]
    Context = context[item['relevant']]
    answer = {"answer_start" : [int(np.int32(item['answer']['start']))], "text" : [item['answer']['text']]} 
    
    validQA.append({
        "answers": answer,
        "context": Context,
        "id": Id,
        "question": question,
        "title": "train"
    })

with open('Data/trainQA.json', 'w', encoding='utf-8') as f:
    json.dump(trainQA, f, ensure_ascii=False, indent=4)
with open('Data/validQA.json', 'w', encoding='utf-8') as f:
    json.dump(validQA, f, ensure_ascii=False, indent=4)


testMC = []
for item in test:

    
    Id = item["id"]
    sent1 = item["question"]
    
    ending = []
    for i in range(0, 4):
        ending.append(context[item['paragraphs'][i]])
    
    testMC.append({
        "id": Id,
        "sent1": sent1,
        "sent2": "",
        "ending0": ending[0],
        "ending1": ending[1],
        "ending2": ending[2],
        "ending3": ending[3],
    })

with open('Data/testMC.json', 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=4)


