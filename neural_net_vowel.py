import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

training_data = pd.read_csv('training_data_vowels.csv')

formants = training_data[["F1", "F2"]].to_numpy()
X_train = (formants - np.mean(formants, axis=0))/np.std(formants, axis=0)
response = training_data["Word"].to_numpy()


color_mapping = {'Bat': 'red', 'Bet': 'orange', 'Beet': 'brown', 'Bit': 'green', 'Bought': 'blue', 'Boot': 'pink'}
response_colors = [color_mapping[response[i]] for i in range(len(response))]

number_mapping = {'Bat': 0, 'Bet': 1, 'Beet': 2, 'Bit': 3, 'Bought': 4, 'Boot': 5}
response_values = [number_mapping[response[i]] for i in range(len(response))]
targets = np.eye(6)[response_values]
print(targets[:5])


#plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(formants[:,1], formants[:,0], c=response_colors, s=50)

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)
ax.set_xlabel("F2 Hz")
ax.set_ylabel("F1 Hz")
ax.set_title("vowel grid")
ax.grid(True)
plt.show()

# --- neural net -----

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(2, 2),
#             nn.ReLU(),
#             nn.Linear(2, 2),
#             nn.ReLU(),
#             nn.Linear(2, 2),
#             nn.ReLU(),
#             nn.Linear(2, 2),
#             nn.ReLU(),
#             nn.Linear(2, 6),
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits
    
#     def train_loop():

 
# model = NeuralNetwork()


            

