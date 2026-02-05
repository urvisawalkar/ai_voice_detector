import base64
import tempfile
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from inference.ensemble_predict import ensemble_predict

API_KEY = "sk_test_123456789"

app = FastAPI(title="Voice Detector API")


class RequestBody(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


def validate_key(key):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/api/voice-detection")
def detect_voice(body: RequestBody, x_api_key: str = Header(...)):

    validate_key(x_api_key)

    if body.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 allowed")

    audio_bytes = base64.b64decode(body.audioBase64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(audio_bytes)
        temp_path = f.name

    label, confidence = ensemble_predict(temp_path)

    os.remove(temp_path)

    return {
        "status": "success",
        "language": body.language,
        "classification": label,
        "confidenceScore": float(confidence),
        "explanation": "Spectral + temporal deep learning ensemble analysis"
    }
