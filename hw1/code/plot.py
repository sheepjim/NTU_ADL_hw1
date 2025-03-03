import numpy as np
import matplotlib.pyplot as plt

with open('./t5_5/report.txt', 'r') as file:
    lines = file.readlines()

rouge1 = np.round(np.array(lines[0].split()).astype(float), decimals=2)
rouge2 = np.round(np.array(lines[1].split()).astype(float), decimals=2)
rougeL = np.round(np.array(lines[2].split()).astype(float), decimals=2)


plt.title('rougeScore')
plt.xlabel('Epoch')
plt.ylabel('rougeScore')
plt.plot(list(range(1, 21)), rouge1, label = 'rouge1')
plt.plot(list(range(1, 21)), rouge2, label = 'rouge2')
plt.plot(list(range(1, 21)), rougeL, label = 'rougeL')
plt.xticks(np.arange(1, 21, step=1))
plt.legend()
plt.savefig('rougeScore.png')