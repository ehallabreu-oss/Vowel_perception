import os
import numpy as np
import pandas as pd
import glob
import random
import soundfile as sf
import sounddevice as sd
import tkinter as tk

# Load files
coords = pd.read_csv("inside_points.csv", skiprows=0).to_numpy()
wav_file = "/Users/Eduardinho/Desktop/UCU/Sem5/Phonetics/vowel recordings/vowel_stims"
wav_paths = sorted(glob.glob(os.path.join(wav_file, "*.wav")))  # finds all wav files in the folder and returns full paths sorted alphabetically

# Create mapping from index to file path (0: /.../stim_002.wav, 1: /.../stim_003.wav, etc.)
mapping = {i: f"/Users/Eduardinho/Desktop/UCU/Sem5/Phonetics/vowel recordings/vowel_stims/stim_{i+2:03d}.wav" for i in range(coords.shape[0])}

# Randomize order
trial_indices = list(mapping.keys()) # create a list of all indices
random.shuffle(trial_indices) # shuffle them randomly
random.seed(0)  # for reproducibility

# play sound
idx = trial_indices[0]  # get index
file_path = mapping[idx]    # get corresponding vowel wav file path
data, sampling_rate = sf.read(file_path) # read the wav file
sd.play(data, sampling_rate) # play the sound
sd.wait()  # Wait until file is done playing

### Simple GUI to display words

choices = ["bat", "bet", "beet", "bit", "bought", "boot"]

def on_click(choice):
    print(f"You selected: {choice}") # callback function

root = tk.Tk() # creates the main window
root.title("Vowel Identification")
label = tk.Label(root,text="Which word did you hear?", font=("Arial", 16))
label.pack(pady=20)

# create buttons for each choice
for word in choices:
    button = tk.Button(root, text=word, font=("Arial", 15), width=10, 
                       command=lambda choice=word: on_click(choice))    #loop thourgh all words, bind each button to its own local variable
    button.pack(pady=5)

root.mainloop()



