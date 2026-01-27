import numpy as np
import pandas as pd
import glob 
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

centers_raw = np.array([
    [267.86, 2363.75],  #beet
    [342.67, 1239.11],  #boot
    [669.84, 995.82],   #bought
    [769.73, 1528.8],   #bat
    [631.82, 1734.60],  #bet
    [461.38, 1906.73]   #bit
]) 

# load data from all participants
csv_files = glob.glob("behavioural_data/*_vowel_classification_results.csv")
data_frames = []
for f in csv_files:
    df = pd.read_csv(f)
    df['source_file'] = f 
    data_frames.append(df)
data = pd.concat(data_frames, ignore_index=False)

count_matrix = data.pivot_table(
    index=['F1', 'F2'], # group all rows with the same F1, F2 combination
    columns='response', #create one column for each new label and for each new file
    aggfunc='size', # count answers
    fill_value=0 # add zeros when no answer
)

prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)


ipa_labels = ["/i/", "/u/", "/ɒ/", "/æ/", "/ɛ/", "/ɪ/"]
word_labels = ["beet", "boot", "bought", "bat", "bet", "bit"]
basic_colors = ['yellow', 'orange', 'red', 'purple', 'blue', 'green']
word_to_color = dict(zip(word_labels, basic_colors))

label_colors = {
    "beet": np.array([1, 1, 0]),    # yellow
    "boot": np.array([1, 0.5, 0]),  # orange
    "bought": np.array([1, 0, 0]),  # red
    "bat": np.array([0.5, 0, 0.5]), # purple  
    "bet": np.array([0, 0, 1]),     # blue
    "bit": np.array([0, 1, 0])      # green
}

rgb_label_matrix = np.vstack(list(label_colors.values()))

# create a DataFrame for rgb values with label index
rgb_df = pd.DataFrame(rgb_label_matrix, index=word_labels, columns=["R","G","B"])

# dot will align columns of prob_mat_ordered with index of rgb_df by label names
color_map_df = prob_matrix.dot(rgb_df)   # DataFrame (N, 3)
color_map = color_map_df.values

# normalizing

def normalize(x):
    return (x - np.mean(x))/np.std(x)

F1_raw = prob_matrix.index.get_level_values('F1').to_numpy()
F2_raw = prob_matrix.index.get_level_values('F2').to_numpy()
F1 = normalize(F1_raw)
F2 = normalize(F2_raw)

F1_center = (centers_raw[:,0] - np.mean(F1_raw))/np.std(F1_raw)
F2_center = (centers_raw[:,1] - np.mean(F2_raw))/np.std(F2_raw)
centers = np.column_stack([F1_center, F2_center])


# Plotting

#plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 5))

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

# synthesized vowels
ax.scatter(F2, F1, c=color_map, s=300)

# natural vowels
for i in range(centers.shape[0]):
    ax.scatter(centers[i,1], centers[i,0], color=basic_colors[i], label=ipa_labels[i], edgecolors='black', s=60)


for (f1, f2), label in zip(centers, ipa_labels):
    ax.text(f2, f1 - 10, label,
            fontsize=14,
            ha='center',  # horizontal alignment
            va='bottom')  # vertical alignment slightly above point 
    
ax.legend(title="Vowel Categories")

ax.set_xlabel("F2 Hz")
ax.set_ylabel("F1 Hz")
ax.set_title("Discrete Perceptual Vowel Map")
plt.show()


# ----------------------
# Interpolated Heat map
# ----------------------

num = 150  # resolution

F1_grid = np.linspace(F1.min(), F1.max(), num)
F2_grid = np.linspace(F2.min(), F2.max(), num)

F2_mesh, F1_mesh = np.meshgrid(F2_grid, F1_grid)
grid_coords = np.column_stack([F2_mesh.ravel(), F1_mesh.ravel()])

points = np.column_stack((F2, F1))

R = color_map[:,0]
G = color_map[:,1]
B = color_map[:,2]

R_grid = griddata(points, R, grid_coords, method='cubic') # (150*150,)
G_grid = griddata(points, G, grid_coords, method='cubic')
B_grid = griddata(points, B, grid_coords, method='cubic')

# reshape into 2d vectors
R_img = R_grid.reshape(F1_mesh.shape) # (150, 150)
G_img = G_grid.reshape(F1_mesh.shape)
B_img = B_grid.reshape(F1_mesh.shape)

RGB_img = np.stack([R_img, G_img, B_img], axis=2) # (150, 150, 3)

# plotting
fig, ax = plt.subplots(figsize=(6, 5))

ax.imshow(
    RGB_img,
    extent=(F2.min(), F2.max(), F1.max(), F1.min()), 
    aspect='auto'
)

ax.scatter(F2, F1, c=color_map, s=50, edgecolors='black')

# natural vowels
# for i in range(centers.shape[0]):
#     ax.scatter(centers[i,1], centers[i,0], color=basic_colors[i], label=ipa_labels[i], edgecolors='black', s=60)

# ax.legend(title="Vowel Categories")

# for (f1, f2), label in zip(centers, ipa_labels):
#     ax.text(f1, f2 - 10, label,
#             fontsize=14,
#             ha='center',  # horizontal alignment
#             va='bottom')  # vertical alignment slightly above point

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

ax.set_xlabel("F2 (Hz)")
ax.set_ylabel("F1 (Hz)")
#ax.set_title("Smooth Perceptual Vowel Map")
plt.show()

# -----------
# 3D Plot
# -----------

print('RGB', RGB_img.shape)

Z = np.max(prob_matrix.to_numpy(), axis=1)
print('Z', Z.shape)

Z_interpolated = griddata(points, Z, grid_coords, method='cubic')

print('Z_interpolated', Z_interpolated.shape)

Z_surf = Z_interpolated.reshape(F1_mesh.shape)
Z_2d = Z_surf.reshape(150, 150)
RGB_surf = np.stack([R_img, G_img, B_img, Z_surf], axis=-1)
RGB_surf_clip = np.clip(RGB_surf, 0, 1)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    F2_mesh, F1_mesh,Z_2d,
    facecolors=RGB_surf_clip,
    rstride=3, cstride=3,
    edgecolor='black',
    linewidth=0.3,
    shade=False
)

hight = np.ones(6)

# for i in range(centers.shape[0]):
#     ax.scatter(centers[i,1], centers[i,0], hight, color=basic_colors[i], label=ipa_labels[i], s=30)

# ax.legend(title="Vowel Categories")

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

ax.set_xlabel("F2 (Hz)")
ax.set_ylabel("F1 (Hz)")
ax.set_zlabel("Certainty (max probability)")

plt.show()

# ----------------------
# Slice up that graph 
# ----------------------

p1 = centers[4]
p2 = centers[1]

num_points = 200
t = np.linspace(0, 1, num_points)
slice_coords = p1 + t[:, None] * (p2 - p1)

# Find closest F1 and F2 values
F1_idx = np.array([np.argmin(np.abs(F1_grid - f1)) for f1 in slice_coords[:,0]])
F2_idx = np.array([np.argmin(np.abs(F2_grid - f2)) for f2 in slice_coords[:,1]])

slice_height_rough = RGB_surf_clip[F1_idx, F2_idx, 3]
slice_height = np.where(np.isnan(slice_height_rough), 1, slice_height_rough)
print(slice_height.shape) # (100,)
slice_rgb = RGB_surf_clip[F1_idx, F2_idx, :3]

degree = 7  # cubic polynomial
coeffs = np.polyfit(t, slice_height, deg=degree)
t_smooth = np.linspace(t.min(), t.max(), 500)
slice_height_poly = np.polyval(coeffs, t_smooth)

plt.figure()
plt.plot(t_smooth, slice_height_poly, color='black', lw=1)
plt.scatter(t, slice_height, c=slice_rgb, s=20)
plt.xlabel('position along line connecting /ɛ/ and /u/')
plt.ylabel('Confidence level')
plt.title('Decision curve between /ɛ/ and /u/')
plt.show()