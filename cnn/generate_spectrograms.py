import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

SR = 16000

def save_specs(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)

    files = [f for f in os.listdir(src_folder) if f.endswith(".mp3") or f.endswith(".wav")]

    for f in tqdm(files):
        path = os.path.join(src_folder, f)

        y, sr = librosa.load(path, sr=SR)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        plt.figure(figsize=(3,3))
        librosa.display.specshow(S_db, sr=sr)
        plt.axis("off")

        save_path = os.path.join(dst_folder, f.replace(".mp3", ".png").replace(".wav",".png"))
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()


import numpy as np

print("Generating HUMAN spectrograms...")
save_specs("../data/human", "spectrograms/human")

print("Generating AI spectrograms...")
save_specs("../data/ai", "spectrograms/ai")

print("Done.")
