"""
Audio-based mental health inference module.

Takes an audio file (minimum 10 seconds), extracts 34-dimensional features
(MFCC + spectral), and returns a mental health score (0-100).

Scoring: negative emotions lower the score, positive/neutral raise it.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.features.audio_features import load_scaler

# Emotional valence mapping: how each emotion maps to mental health (0=worst, 100=best)
EMOTION_HEALTH_MAP = {
    "happy": 95,
    "neutral": 75,
    "calm": 85,
    "surprise": 60,
    "sad": 20,
    "angry": 30,
    "fear": 15,
    "disgust": 25,
}

_model_cache = {}


def load_model(cfg: dict = None):
    """Load the trained audio model (cached)."""
    cfg = cfg or config.AUDIO_CONFIG
    key = cfg["model_path"]
    if key not in _model_cache:
        if not os.path.exists(cfg["model_path"]):
            raise FileNotFoundError(
                f"Audio model not found at {cfg['model_path']}. "
                "Run: python train_all.py"
            )
        _model_cache[key] = joblib.load(cfg["model_path"])
    return _model_cache[key]


def get_audio_duration(file_path: str, sr: int = 22050) -> float:
    """Get audio duration in seconds."""
    import librosa
    y, _ = librosa.load(file_path, sr=sr)
    return len(y) / sr


def predict_score(audio_path: str, cfg: dict = None) -> dict:
    """
    Predict mental health score from an audio file.

    Requires audio length between min_duration_sec and max_duration_sec (from config).

    Returns dict with score, confidence, predicted emotion, probabilities.
    """
    cfg = cfg or config.AUDIO_CONFIG

    if not cfg["enabled"]:
        return {"score": None, "predicted_emotion": None, "available": False}

    from src.data.audio_loader import extract_audio_features

    # Check duration
    duration = get_audio_duration(audio_path, sr=cfg["sample_rate"])
    min_dur = cfg.get("min_duration_sec", 10)
    if duration < min_dur:
        return {
            "score": None,
            "available": False,
            "error": f"Audio too short: {duration:.1f}s (minimum {min_dur}s required)",
            "duration": round(duration, 1),
        }

    max_dur = cfg.get("max_duration_sec")
    if max_dur and duration > max_dur:
        return {
            "score": None,
            "available": False,
            "error": f"Audio too long: {duration:.1f}s (maximum {max_dur}s)",
            "duration": round(duration, 1),
        }

    model = load_model(cfg)
    scaler = load_scaler(cfg)

    feats = extract_audio_features(audio_path, sr=cfg["sample_rate"], n_mfcc=cfg["n_mfcc"])
    X = scaler.transform(feats.reshape(1, -1))

    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    # Decode class labels
    le_path = os.path.join(config.MODELS_DIR, "audio_label_encoder.joblib")
    if os.path.exists(le_path):
        le = joblib.load(le_path)
        class_names = list(le.inverse_transform(classes))
    else:
        class_names = [str(c) for c in classes]

    # Compute weighted health score from emotion probabilities
    score = 0.0
    for i, name in enumerate(class_names):
        health_val = EMOTION_HEALTH_MAP.get(name.lower(), 50)
        score += proba[i] * health_val

    score = round(max(0.0, min(100.0, score)), 2)
    confidence = round(float(max(proba)), 4)
    predicted_idx = int(np.argmax(proba))
    predicted_emotion = class_names[predicted_idx]

    emotion_probs = {class_names[i]: round(float(proba[i]), 4) for i in range(len(class_names))}

    return {
        "score": score,
        "confidence": confidence,
        "predicted_emotion": predicted_emotion,
        "emotion_probabilities": emotion_probs,
        "duration": round(duration, 1),
        "available": True,
    }


if __name__ == "__main__":
    print("Audio inference ready. Use predict_score(audio_path) with a 10+ second .wav file.")
