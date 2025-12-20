import os
import numpy as np
import pandas as pd
import glob
import random
import soundfile as sf
import sounddevice as sd
import tkinter as tk
from tkinter import simpledialog

# Load files
coords = pd.read_csv("inside_points.csv", skiprows=0).to_numpy()
wav_file = "vowel_stims"
wav_paths = sorted(glob.glob(os.path.join(wav_file, "*.wav")))  # finds all wav files in the folder and returns full paths sorted alphabetically

# Create mapping from index to file path (0: /.../stim_002.wav, 1: /.../stim_003.wav, etc.)
mapping = {i: f"vowel_stims/stim_{i+2:03d}.wav" for i in range(coords.shape[0])}

# Randomize order
trial_indices = list(mapping.keys()) # create a list of all indices
random.shuffle(trial_indices) # shuffle them randomly

# intialize variables
choices = ["bat", "bet", "beet", "bit", "bought", "boot"]
results = []

# ---------------------
# Experiment class
# ---------------------

class Experiment:
    def __init__(self, root):
        self.root = root       
        self.participant_id = participant_id

        # --- Start window ---
        self.start_label = tk.Label(root, text="Welcome to this experiment!", font=("Arial", 36))
        self.start_label.pack(pady=30)  
        self.instructions = tk.Label(root, text="You will hear a word.\n" \
        "Choose the word you think it is from the 6 options bellow.\n\n"
        "Bat\n" "Bet\n" "Beet\n" "Bit\n" "Bought\n" "Boot\n\n"
        "Press SPACE to begin", font=("Arial", 34))
        self.instructions.pack(pady=20)

        self.root.bind("<space>", self.start_experiment) # bind space key
  
        self.current_trial = -1 #so first increment lands at 0
        self.started = False  

    def start_experiment(self, event=None):
        if self.started: # if it started, space bar does nothing
            return None
        self.started = True

        # clears initial instructrions
        self.start_label.destroy()
        self.instructions.destroy()
        self.root.unbind("<space>")

        # between trial labels
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
        data, sampling_rate = sf.read(file_path) 
        sd.play(data, sampling_rate) 

        #remember which stim was played
        self.current_idx = idx

    def record_answer(self, choice): 
        F1, F2 = coords[self.current_idx]
        
        results.append({
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
        self.root.after(700, self.next_trial)
        
    def end_experiment(self):
        df = pd.DataFrame(results)
        file_name = f"{self.participant_id}_vowel_classification_results.csv"
        df.to_csv(file_name, index=False)
        print(df)
        self.root.quit() #close window

# ---------------
# Run Experiment
# ---------------

root = tk.Tk() # create main window
root.geometry("900x800")
root.withdraw()
participant_id = simpledialog.askstring("Participant ID", "Enter your participant ID:")
root.deiconify()
Experiment(root) # create experiment object inside main window
root.mainloop() # start the GUI event loop