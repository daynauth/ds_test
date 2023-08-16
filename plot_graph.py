import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('./benchmark/results.csv')


df.plot(x="model", y=["deepspeed","huggingface"], kind="bar", title="DeepSpeed")

plt.xticks(rotation=0)
plt.xlabel("Model")
plt.ylabel("Inference (ms)")

plt.savefig('deepseed.pdf')
