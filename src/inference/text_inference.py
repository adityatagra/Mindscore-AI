"""
Text-based mental health inference module.

Takes raw text, preprocesses it, runs through the trained ensemble model,
and returns a mental health score (0-100) with confidence.

Scoring:
  score = (1 - P(distress)) * 100
  confidence = max(P(distress), P(normal))  -- how certain the model is
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.text_loader import clean_text
from src.features.text_features import load_vectorizer

_model_cache = {}


def load_model(cfg: dict = None):
    """Load the trained text model (cached after first load)."""
    cfg = cfg or config.TEXT_CONFIG
    cache_key = cfg["model_path"]
    if cache_key not in _model_cache:
        if not os.path.exists(cfg["model_path"]):
            raise FileNotFoundError(
                f"Text model not found at {cfg['model_path']}. "
                "Run: python train_all.py"
            )
        _model_cache[cache_key] = joblib.load(cfg["model_path"])
    return _model_cache[cache_key]


def predict_score(text: str, cfg: dict = None) -> dict:
    """
    Predict mental health score from a single text input.

    Returns dict with score (0-100), confidence, distress probability, and class.
    """
    cfg = cfg or config.TEXT_CONFIG
    model = load_model(cfg)
    vectorizer = load_vectorizer(cfg)

    cleaned = clean_text(text)
    if not cleaned.strip():
        return {
            "score": 50.0,
            "confidence": 0.0,
            "distress_probability": 0.5,
            "predicted_class": 0,
            "input_quality": "empty",
        }

    word_count = len(cleaned.split())
    input_quality = "good" if word_count >= 10 else ("fair" if word_count >= 4 else "poor")

    X = vectorizer.transform([cleaned])
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)

    distress_idx = classes.index(1) if 1 in classes else -1
    if distress_idx >= 0:
        p_distress = float(proba[distress_idx])
    else:
        p_distress = 0.5

    score = round((1.0 - p_distress) * 100, 2)
    score = max(0.0, min(100.0, score))
    confidence = round(float(max(proba)), 4)

    return {
        "score": score,
        "confidence": confidence,
        "distress_probability": round(p_distress, 4),
        "predicted_class": int(model.predict(X)[0]),
        "predicted_label": "distress" if model.predict(X)[0] == 1 else "normal",
        "input_quality": input_quality,
        "word_count": word_count,
    }


if __name__ == "__main__":
    samples = [
        "I feel great today, had a wonderful time with friends and family. Life is good.",
        "I can't stop crying. Everything feels hopeless and I don't want to get out of bed. Nothing matters anymore.",
        "Work was okay today. Nothing special happened but nothing bad either.",
    ]
    for t in samples:
        r = predict_score(t)
        print(f"Score: {r['score']:5.1f} | Confidence: {r['confidence']:.2f} | "
              f"Class: {r['predicted_label']:<8s} | Quality: {r['input_quality']}")
        print(f"  Text: {t[:80]}...\n")
