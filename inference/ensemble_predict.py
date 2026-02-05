import os
import joblib
import librosa
import numpy as np

# ======================
# PATH SETUP (Render safe)
# ======================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XGB_PATH = os.path.join(BASE, "models", "forensic_model.pkl")


# ======================
# LOAD MODEL (FAST)
# ======================
xgb_model = joblib.load(XGB_PATH)
print("✅ XGBoost model loaded (fast mode)")


# ======================
# FEATURE EXTRACTION
# ======================
def extract_features(path):
    y, sr = librosa.load(path, sr=16000)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    spec = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    return [mfcc, zcr, rms, spec]


# ======================
# FAST PREDICT (XGB ONLY)
# ======================
def ensemble_predict(path):
    """
    FAST VERSION for cloud deployment.
    Uses only XGBoost to avoid timeout on Render free tier.
    Response time: ~0.5–1 second
    """

    feats = extract_features(path)

    prob = xgb_model.predict_proba([feats])[0][1]

    label = "AI_GENERATED" if prob >= 0.65 else "HUMAN"

    return label, float(prob)


# ======================
# CLI TEST (optional)
# ======================
if __name__ == "__main__":
    import sys
    audio = sys.argv[1]
    label, conf = ensemble_predict(audio)
    print(label, conf)
