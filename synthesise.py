import os
import numpy as np
import pandas as pd
import parselmouth
from scipy import signal  # Ensure this import is correct
from scipy.io import wavfile

template_path = "/Users/Eduardinho/Desktop/UCU/Sem5/Phonetics/vowel recordings/synthesised vowels/b_t.wav"
sound = parselmouth.Sound(template_path)

csv_path = "inside_points.csv"
targets = pd.read_csv(csv_path, sep=",")
output_dir = "/Users/Eduardinho/Desktop/UCU/Sem5/Phonetics/vowel recordings/synthesised vowels"
os.makedirs(output_dir, exist_ok=True)

def estimate_F3_from_spectrum(sound, low_hz=1800, high_hz=4000):
    sig = sound.values.flatten()
    sr = sound.sampling_frequency
    N = len(sig)
    if N < 2:
        return 3000.0
    w = np.hamming(N)
    S = np.abs(np.fft.rfft(sig * w))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return 3000.0
    S_band = S[mask]
    freqs_band = freqs[mask]
    idx = np.argmax(S_band)
    return float(freqs_band[idx])

def make_glottal_pulse(f0, duration, sr):
    """Generate a glottal pulse train."""
    period = int(sr / f0)
    pulse = np.zeros(period)
    pulse[0] = 1  # Simple impulse
    pulse = signal.lfilter([1], [1, -0.9], pulse)  # Apply a simple decay
    pulse = pulse / np.max(np.abs(pulse))  # Normalize
    num_pulses = int(np.ceil(duration * f0))
    return np.tile(pulse, num_pulses)[:int(duration * sr)]

def bandpass_filter(signal_data, sr, center_freq, bandwidth):
    """Apply a bandpass filter to the signal."""
    nyquist = 0.5 * sr
    low = center_freq - bandwidth / 2
    high = center_freq + bandwidth / 2
    low = max(1.0, low) / nyquist
    high = min(nyquist - 1.0, high) / nyquist
    b, a = signal.butter(2, [low, high], btype='band')
    return signal.lfilter(b, a, signal_data)

def synthesize_vowel(F1, F2, F3, f0, duration, sr):
    """Synthesize a vowel using a glottal pulse and formant filtering."""
    glottal_pulse = make_glottal_pulse(f0, duration, sr)
    
    # Apply bandpass filters for each formant
    filtered_signal = bandpass_filter(glottal_pulse, sr, F1, 60)  # F1
    filtered_signal = bandpass_filter(filtered_signal, sr, F2, 90)  # F2
    filtered_signal = bandpass_filter(filtered_signal, sr, F3, 150)  # F3

    # Normalize the output
    if np.max(np.abs(filtered_signal)) > 0:
        filtered_signal = filtered_signal / np.max(np.abs(filtered_signal)) * 0.95
    return filtered_signal

# Prepare to synthesize
measured_F3 = estimate_F3_from_spectrum(sound)

# Ensure CSV has F1 and F2 columns (case-insensitive)
cols = {c.lower(): c for c in targets.columns}
if 'f1' not in cols or 'f2' not in cols:
    raise KeyError(f"CSV must contain F1 and F2 columns. Found: {targets.columns.tolist()}")

for i, row in targets.iterrows():
    F1 = float(row[cols['f1']])
    F2 = float(row[cols['f2']])
    f0 = 120.0
    duration = sound.get_total_duration()
    out_sig = synthesize_vowel(F1, F2, measured_F3, f0, duration, sound.sampling_frequency)

    # ensure sample rate is an integer
    sr = int(round(sound.sampling_frequency))

    # make sure output is a 1-D int16 numpy array
    wav_data = np.asarray(out_sig)
    if wav_data.ndim > 1:
        wav_data = np.squeeze(wav_data)
    wav_int16 = np.int16(np.clip(wav_data * 32767.0, -32767, 32767))

    filename = f"vowel_F1_{int(round(F1))}_F2_{int(round(F2))}.wav"
    filepath = os.path.join(output_dir, filename)
    wavfile.write(filepath, sr, wav_int16)
    print("Saved", filepath)

