import matplotlib.pyplot as plt

with open("model_70", "r",encoding="utf-8") as f:
    lines = f.readlines()

ppl = []
for step, line in enumerate(lines, 1):
    if step % 2 == 0:
        ppl.append(float(line.split(": ")[1][:-1]))

plt.plot(list(range(0, len(ppl))), ppl, label = 'Learning Curve')
plt.title('Learning Curve')
plt.xlabel("Steps")
plt.ylabel("meanPerplexity")
plt.savefig('./meanPerplexity.png')

print(min(ppl))