import os
import numpy as np
import pandas as pd
import parselmouth
from tqdm import tqdm


# ============================================
# Function: extract forensic features
# ============================================
def extract_features(path):
    snd = parselmouth.Sound(path)

    # ---- Pitch standard deviation ----
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]

    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0


    # ---- Jitter ----
    point = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )

    jitter = parselmouth.praat.call(
        point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )


    # ---- Shimmer ----
    shimmer = parselmouth.praat.call(
        [snd, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )


    # ---- Harmonic-to-noise ratio ----
    harmonicity = parselmouth.praat.call(
        snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
    )

    hnr = parselmouth.praat.call(
        harmonicity, "Get mean", 0, 0
    )


    return [pitch_std, jitter, shimmer, hnr]


# ============================================
# Main script: process all files
# ============================================

data = []

folders = [
    ("../data/human", 0),   # label 0 = human
    ("../data/ai", 1)      # label 1 = AI
]

for folder, label in folders:

    files = [f for f in os.listdir(folder) if f.endswith(".mp3")]

    for file in tqdm(files, desc=f"Processing {folder}"):

        path = os.path.join(folder, file)

        feats = extract_features(path)
        feats.append(label)

        data.append(feats)


# ============================================
# Save CSV
# ============================================

columns = ["pitch_std", "jitter", "shimmer", "hnr", "label"]

df = pd.DataFrame(data, columns=columns)

df.to_csv("../data/features.csv", index=False)

print("\nDone! features.csv created inside data/")
print(df.head())

