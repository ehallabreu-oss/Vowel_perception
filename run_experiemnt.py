import os
import numpy as np
import pandas as pd
import glob
import random
import soundfile as sf


# Load files
coords = pd.read_csv("inside_points.csv", skiprows=0).to_numpy()
wav_file = "/Users/Eduardinho/Desktop/UCU/Sem5/Phonetics/vowel recordings/vowel_stims"
wav_paths = sorted(glob.glob(os.path.join(wav_file, "*.wav")))  # finds all wav files in the folder and returns full paths sorted aplphabetically

# Create mapping from index to file path (0: /.../stim_002.wav, 1: /.../stim_003.wav, etc.)
mapping = {i: f"Users/Eduardinho/Desktop/UCU/Sem5/Phonetics/vowel recordings/vowel_stims/stim_{i+2:03d}.wav" for i in range(coords.shape[0])}

# Randomize order
trial_indices = list(mapping.keys())
random.shuffle(trial_indices)



