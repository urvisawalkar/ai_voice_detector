import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import random
import numpy as np   # ⭐ important

# =====================================================
# Choose folder
# =====================================================
folder = "../data/human"   # change to ../data/ai to test AI


# =====================================================
# Auto pick random mp3
# =====================================================
files = [f for f in os.listdir(folder) if f.lower().endswith(".mp3")]

FILE = os.path.join(folder, random.choice(files))

print("\nLoading:", FILE)


# =====================================================
# Load audio
# =====================================================
y, sr = librosa.load(FILE, sr=16000)

print("Duration:", round(len(y)/sr, 2), "seconds")
print("Sample rate:", sr)


# =====================================================
# Plot everything in ONE window
# =====================================================
fig, ax = plt.subplots(3, 1, figsize=(10, 8))


# ---------- Waveform ----------
librosa.display.waveshow(y, sr=sr, ax=ax[0])
ax[0].set_title("Waveform")


# ---------- Spectrogram ----------
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

S_db = librosa.power_to_db(S, ref=np.max)   # ⭐ fixed here

img = librosa.display.specshow(
    S_db,
    sr=sr,
    x_axis="time",
    y_axis="mel",
    ax=ax[1]
)

fig.colorbar(img, ax=ax[1])
ax[1].set_title("Mel Spectrogram")


# ---------- Pitch ----------
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
pitch_values = pitches[magnitudes > 0]

ax[2].plot(pitch_values)
ax[2].set_title("Pitch Contour")


plt.tight_layout()
plt.show()
