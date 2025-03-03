import json
import numpy as np
import torch
import pandas as pd
import sys
from transformers import AutoModelForMultipleChoice, AutoTokenizer, pipeline

with open(sys.argv[1], 'r', encoding="utf-8") as f:
    context = json.load(f)
with open(sys.argv[2], 'r', encoding='utf-8') as f:
    test = json.load(f)

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

MCmodel = AutoModelForMultipleChoice.from_pretrained("./modelMC")
MCtokenizer = AutoTokenizer.from_pretrained("./modelMC")

ids = []
questions = []
choices = []
for item in testMC:
    ids.append(item['id'])
    questions.append(item['sent1'])
    choice = []
    for i in range(0, 4):
        choice.append(item[f'ending{i}']) 
    choices.append(choice)

res = {"id" : [], "answer" : []}
for id, question, choice in zip(ids, questions, choices):
    inputs = MCtokenizer([[question, choice[0]], [question, choice[1]], [question, choice[2]], [question, choice[3]]] , padding = True, truncation = True, max_length = 512, return_tensors="pt")
    with torch.no_grad():
        outputs = MCmodel(**{k: v.unsqueeze(0) for k, v in inputs.items()})
    logits = outputs.logits
    prediction = logits.argmax().item()
    
    sent = choice[prediction]
    QA = pipeline("question-answering", model = "./modelQA", device = 0)
    answer = QA(question = question, context = sent)
    res['id'].append(id)
    res['answer'].append(answer['answer'])    

df = pd.DataFrame(res)
df.to_csv(sys.argv[3], index=False)
