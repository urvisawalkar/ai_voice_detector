import os
import torch
import librosa
import joblib
import numpy as np
from torchvision import models
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(__file__))

XGB_PATH = os.path.join(BASE, "models", "forensic_model.pkl")
CNN_PATH = os.path.join(BASE, "cnn", "cnn_model.pth")

# ======================
# LOAD MODELS
# ======================
xgb_model = joblib.load(XGB_PATH)

cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(512, 2)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
cnn_model.eval()

print("✅ Models loaded")


# ======================
# FEATURE EXTRACT
# ======================
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spec = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return [mfcc, zcr, rms, spec]


# ======================
# XGB (probability)
# ======================
def xgb_predict(path):
    feats = extract_features(path)
    prob = xgb_model.predict_proba([feats])[0][1]
    return float(prob)


# ======================
# CNN (probability)
# ======================
def cnn_predict(path):
    y, sr = librosa.load(path, sr=16000)

    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec = librosa.power_to_db(spec)

    spec = torch.tensor(spec).float()
    spec = (spec - spec.mean()) / spec.std()

    spec = spec.unsqueeze(0)
    spec = spec.repeat(3, 1, 1)
    spec = spec.unsqueeze(0)

    with torch.no_grad():
        out = cnn_model(spec)
        prob = torch.softmax(out, dim=1)[0][1].item()

    return float(prob)


# ======================
# ENSEMBLE (FINAL)
# ======================
def ensemble_predict(path):

    p1 = float(xgb_predict(path))
    p2 = float(cnn_predict(path))

    score = 0.3 * p1 + 0.7 * p2

    label = "AI_GENERATED" if score >= 0.65 else "HUMAN"

    return label, float(score)


# ======================
# CLI TEST
# ======================
if __name__ == "__main__":
    import sys
    audio = sys.argv[1]
    label, conf = ensemble_predict(audio)
    print(label, conf)
