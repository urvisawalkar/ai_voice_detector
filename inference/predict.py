import torch
import librosa
import sys
from torchvision import models
import torch.nn as nn

# -------------------------
# Load CNN
# -------------------------
cnn_model = models.resnet18(weights=None)
cnn_model.fc = nn.Linear(512, 2)

state = torch.load("../cnn/cnn_model.pth", map_location="cpu")
cnn_model.load_state_dict(state)

cnn_model.eval()

print("✅ CNN loaded")


# -------------------------
# Predict
# -------------------------
def predict(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)

    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec = librosa.power_to_db(spec)

    spec = torch.tensor(spec).float()

    spec = (spec - spec.mean()) / spec.std()

    spec = spec.unsqueeze(0)      # (1,H,W)
    spec = spec.repeat(3, 1, 1)   # 🔥 make 3 channel
    spec = spec.unsqueeze(0)      # batch

    with torch.no_grad():
        out = cnn_model(spec)
        pred = torch.argmax(out, 1).item()

    return pred


audio = sys.argv[1]
p = predict(audio)

print("Prediction:", "AI" if p == 1 else "Human")
