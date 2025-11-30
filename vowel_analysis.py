import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

centers = np.array([
    [769.73, 1528.8],  
    [631.82, 1734.60],
    [267.86, 2363.75],
    [461.38, 1906.73],
    [669.84, 995.82],
    [342.67, 1239.11]
]) 
labels = ["/æ/", "/ɛ/", "/i/", "/ɪ/", "/ɒ/", "/u/"]
basic_colors = ['red', 'orange', 'brown', 'green', 'blue', 'pink']

data = pd.read_csv("Avery_3_vowel_classification_results.csv")
formants = data[["F1", "F2"]].to_numpy()
response = data["response"].to_numpy()

color_mapping = {'bat': 'red', 'bet': 'orange', 'beet': 'brown', 'bit': 'green', 'bought': 'blue', 'boot': 'pink'}
response_dict = {i: response for i, response in enumerate(response)}
response_colors = [color_mapping[response_dict[i]] for i in range(len(response_dict))]

fig, ax = plt.subplots(figsize=(6, 5))
ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

for idx in range(len(response_colors)):
    ax.scatter(formants[idx,1], formants[idx,0], color=response_colors[idx], s=50)

for i in range(centers.shape[0]):
    ax.scatter(centers[i,1], centers[i,0], color=basic_colors[i], label=labels[i], edgecolors='black', s=20)

ax.legend(title="Vowel Categories")

for (F1, F2), label in zip(centers, labels):
    ax.text(F2, F1 - 10, label, fontsize=14, ha='center', va='bottom')  

ax.set_xlabel("F2 Hz")
ax.set_ylabel("F1 Hz")
ax.set_title("vowel grid")
plt.show()