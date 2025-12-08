import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from group_analysis import RGB_img
from scipy.spatial import ConvexHull

df = pd.read_csv('Training_data_participants.csv', sep=';')

df['F1_norm'] = df.groupby('Participant')['F1'].transform(lambda x: (x - x.mean()) / x.std())
df['F2_norm'] = df.groupby('Participant')['F2'].transform(lambda x: (x - x.mean()) / x.std())

formants = df[["F1_norm", "F2_norm"]].to_numpy()
word = df["Word"].to_numpy()

idx = np.random.permutation(len(formants))
X_shuffled = formants[idx] 
response = word[idx]      

color_mapping = {'Beet': 'yellow','Boot': 'orange','Bought': 'red', 'Bat': 'purple', 'Bet': 'blue', 'Bit': 'lime'}
response_colors = [color_mapping[response[i]] for i in range(len(response))]

number_mapping = {'Beet': 0, 'Boot': 1, 'Bought': 2, 'Bat': 3, 'Bet': 4, 'Bit': 5}
response_values = [number_mapping[response[i]] for i in range(len(response))]

X_train = torch.tensor(X_shuffled[:100], dtype=torch.float32)
y_train = torch.tensor(response_values[:100], dtype=torch.long)

X_test = torch.tensor(X_shuffled[100:], dtype=torch.float32)
y_test = torch.tensor(response_values[100:], dtype=torch.long)

# --- neural net -----
TRAIN = False

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 2),
            nn.Tanh(),
            nn.Linear(2, 6),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
 
model = NeuralNetwork()

if TRAIN:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.01)

    epochs = 4000

    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            pred = logits.argmax(dim=-1)
            accuracy = (pred == y_train).float().mean().item()
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Accuracy: {accuracy:.3f}")

    torch.save(model.state_dict(), 'vowel_classification_model.pth')

model.load_state_dict(torch.load('vowel_classification_model.pth'))
model.eval()

with torch.no_grad():   # no gradients needed
    test_logits = model(X_test)
    test_pred = test_logits.argmax(dim=1)
    test_accuracy = (test_pred == y_test).float().mean().item()

print(f"Test Accuracy: {test_accuracy:.3f}")

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 5))

ax.scatter(X_shuffled[:,1], X_shuffled[:,0], c=response_colors, s=50)

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)
ax.set_xlabel("F2 Hz")
ax.set_ylabel("F1 Hz")
ax.set_title("vowel grid")
ax.grid(True)
plt.show()

# ----------------------
# Decision heat map
# ----------------------

num = 150  # resolution

F1_norm = df['F1_norm'].to_numpy()
F2_norm = df['F2_norm'].to_numpy()

F1_grid = np.linspace(F1_norm.min(), F1_norm.max(), num)
F2_grid = np.linspace(F2_norm.min(), F2_norm.max(), num)

F2_mesh, F1_mesh = np.meshgrid(F1_grid, F2_grid)
grid_coords = np.column_stack([F1_mesh.ravel(), F2_mesh.ravel()])

grid_tensor = torch.tensor(grid_coords, dtype=torch.float32)

with torch.no_grad():
    logits = model(grid_tensor)
    pred_matrix = torch.softmax(logits, dim=1).numpy()

label_colors = {
    "beet": np.array([1, 1, 0]),    # yellow
    "boot": np.array([1, 0.5, 0]),  # orange
    "bought": np.array([1, 0, 0]),  # red
    "bat": np.array([0.5, 0, 0.5]), # purple  
    "bet": np.array([0, 0, 1]),     # blue
    "bit": np.array([0, 1, 0])      # green
}

rgb_label_matrix = np.vstack(list(label_colors.values()))
RGB_matrix = pred_matrix @ rgb_label_matrix # (22500, 3)
RGB_ann = RGB_matrix.reshape(num, num, 3)

threshold = 0.001
behav_mask = np.any(RGB_img > threshold, axis=-1) # is any of the 3 channels non zero? 
print('behav_mask', behav_mask.shape)
RGB_ann_masked = RGB_ann.copy()
RGB_ann_surf = RGB_ann.copy()
RGB_ann_masked[~behav_mask] = [0, 0, 0]
RGB_ann_surf[~behav_mask] = [1, 1, 1]

print("behav zeros per channel:", (RGB_img == 0).all(axis=(0,1)))

fig, ax = plt.subplots(figsize=(6, 5))

ax.imshow(
    RGB_ann_masked,
    extent=(F2_norm.min(), F2_norm.max(), 
            F1_norm.max(), F1_norm.min()), 
    aspect='auto'
)
ax.scatter(X_shuffled[:,1], X_shuffled[:,0], c=response_colors, s=50, edgecolor='black')


ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

ax.set_xlabel("F2 (Hz)")
ax.set_ylabel("F1 (Hz)")
ax.set_title("Neural Net Decision Map")
plt.show()

# -------
# 3D Plot
# -------

Z = np.max(pred_matrix, axis=1)
print(Z.shape)
Z_2d = Z.reshape(150, 150)
Z_2d_masked = Z_2d.copy()
Z_2d_masked[~behav_mask] = 0

RGB_surf = np.stack([RGB_ann_masked[:,:,0], RGB_ann_masked[:,:,1], RGB_ann_masked[:,:,2], Z_2d], axis=-1)
RGB_surf_clip = np.clip(RGB_surf, 0, 1)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    F2_mesh,
    F1_mesh,
    Z_2d_masked,
    facecolors=RGB_surf_clip,
    rstride=3, cstride=3,
    edgecolor='black',
    linewidth=0,
    antialiased=False,
    shade=False
)

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

ax.set_xlabel("F2 (Hz)")
ax.set_ylabel("F1 (Hz)")
ax.set_zlabel("Certainty (max probability)")

plt.show()








