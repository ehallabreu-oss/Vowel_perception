import numpy as np
import pandas as pd
import glob 
import matplotlib.pyplot as plt

centers = np.array([
    [769.73, 1528.8],  
    [631.82, 1734.60],
    [267.86, 2363.75],
    [461.38, 1906.73],
    [669.84, 995.82],
    [342.67, 1239.11]
]) 

# load data from all participants
csv_files = glob.glob("*_vowel_classification_results.csv")
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


ipa_labels = ["/æ/", "/ɛ/", "/i/", "/ɪ/", "/ɒ/", "/u/"]
word_labels = ["bat", "bet", "beet", "bit", "bought", "boot"]
basic_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
word_to_color = dict(zip(word_labels, basic_colors))

label_colors = {
    "bat": np.array([1, 0, 0]), 
    "bet": np.array([1, 0.5, 0]),
    "beet": np.array([1, 1, 0]),
    "bit": np.array([0, 1, 0]),
    "bought": np.array([0, 0, 1]),
    "boot": np.array([0.5, 0, 0.5])
}

rgb_label_matrix = np.vstack(list(label_colors.values()))

# create a DataFrame for rgb values with label index
rgb_df = pd.DataFrame(rgb_label_matrix, index=word_labels, columns=["R","G","B"])

# dot will align columns of prob_mat_ordered with index of rgb_df by label names
color_map_df = prob_matrix.dot(rgb_df)   # DataFrame (N, 3)
color_map = color_map_df.values


# prob_matrix['predicted_word'] = prob_matrix.idxmax(axis=1)
# prob_matrix['color'] = prob_matrix['predicted_word'].map(word_to_color)
# color_list = prob_matrix['color'] 

F1 = prob_matrix.index.get_level_values('F1')
F2 = prob_matrix.index.get_level_values('F2') 


# Plotting
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 5))

ax.xaxis.set_inverted(True)
ax.yaxis.set_inverted(True)

# synthesized vowels
ax.scatter(F2, F1, c=color_map, s=300)

# natural vowels
for i in range(centers.shape[0]):
    ax.scatter(centers[i,1], centers[i,0], color=basic_colors[i], label=ipa_labels[i], edgecolors='black', s=60)

ax.legend(title="Vowel Categories")

for (F1, F2), label in zip(centers, ipa_labels):
    ax.text(F2, F1 - 10, label, fontsize=14, ha='center', va='bottom')  

ax.set_xlabel("F2 Hz")
ax.set_ylabel("F1 Hz")
ax.set_title("vowel grid")
plt.show()

