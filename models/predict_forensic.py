import joblib
import numpy as np
import parselmouth
import sys


# ============================================
# Load trained model
# ============================================
model = joblib.load("forensic_model.pkl")


# ============================================
# Feature extraction (same as before)
# ============================================
def extract_features(path):
    snd = parselmouth.Sound(path)

    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0

    point = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )

    jitter = parselmouth.praat.call(
        point, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )

    shimmer = parselmouth.praat.call(
        [snd, point], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    harmonicity = parselmouth.praat.call(
        snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0
    )

    hnr = parselmouth.praat.call(
        harmonicity, "Get mean", 0, 0
    )

    return [pitch_std, jitter, shimmer, hnr]


# ============================================
# Predict
# ============================================
audio_path = sys.argv[1]

features = extract_features(audio_path)
features = np.array(features).reshape(1, -1)

pred = model.predict(features)[0]

label = "AI GENERATED" if pred == 1 else "HUMAN"

print("\nPrediction:", label)
