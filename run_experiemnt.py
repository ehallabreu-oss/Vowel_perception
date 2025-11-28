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

# intialize variables
choices = ["bat", "bet", "beet", "bit", "bought", "boot"]
results = []

# ----------------------
# Main experiment class
# ----------------------

class Experiment:
    def __init__(self, root):
        self.root = root       
        self.label = tk.Label(root, text="Which word did you hear?", font=("Arial", 28))
        self.label.pack(pady=20) #add to the window with vertical padding
        self.message_label = tk.Label(root, text="", font=("Arial", 28))
        self.message_label.pack(pady=30)
        
        self.buttons = []
        for word in choices:
            button = tk.Button(root, text=word, font=("Arial", 24), width=12, height=2,
                               command=lambda choice=word: self.record_answer(choice))
            button.pack(pady=10)
            self.buttons.append(button)

        self.current_trial = -1 #so first increment lands at 0
        self.next_trial()

    def next_trial(self):
        self.message_label.config(text="") # clear message
        self.root.update_idletasks() # ensure clear message is drawn before playing
        for button in self.buttons:
            button.config(state="normal")
        self.current_trial += 1 # move to next trial

        # end experiment if no more trials
        if self.current_trial >= len(trial_indices):
            print("Experiment finished!")
            self.end_experiment()
            return None

        # play sound
        idx = trial_indices[self.current_trial] # get index of current trial
        file_path = mapping[idx]    # get corresponding vowel wav file path
        data, sampling_rate = sf.read(file_path) # read the wav file
        sd.play(data, sampling_rate) # play the sound
        sd.wait()  # Wait until file is done playing

        #remember which stim was played
        self.current_idx = idx

    def record_answer(self, choice): # record answer per trial
        F1, F2 = coords[self.current_idx]
        
        results.append({
            "idx": self.current_idx,
            # "wav_file": mapping[self.current_idx],
            "index": self.current_idx,
            "F1": F1,
            "F2": F2,
            "response": choice,
            "bat": 1 if choice == "bat" else 0,
            "bet": 1 if choice == "bet" else 0,
            "beet": 1 if choice == "beet" else 0,
            "bit": 1 if choice == "bit" else 0,
            "bought": 1 if choice == "bought" else 0,
            "boot": 1 if choice == "boot" else 0,
        })

        self.message_label.config(text="Next word...")
        self.root.update_idletasks()
        for button in self.buttons:
            button.config(state="disabled")
        self.root.after(1000, self.next_trial)
        

    def end_experiment(self):
        df = pd.DataFrame(results)
        df.to_csv("vowel_classification_results.csv", index=False)
        print(df)
        self.root.quit() #close window

# ---------------
# Run Experiment
# ---------------

root = tk.Tk() # create main window
root.geometry("400x800")
Experiment(root) # create experiment object inside main window
root.mainloop() # start the GUI event loop