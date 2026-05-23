"""
Questionnaire-based mental health inference module.

Takes questionnaire answers (numeric features), runs through the trained
ensemble model, and returns a mental health score (0-100) with confidence.

Scoring:
  score = P(low_risk) * 100 + P(moderate_risk) * 50 + P(high_risk) * 0
  confidence = max probability among classes
"""

import os
import sys
import json
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.features.questionnaire_features import load_scaler

_model_cache = {}


def load_model(cfg: dict = None):
    """Load the trained questionnaire model (cached)."""
    cfg = cfg or config.QUESTIONNAIRE_CONFIG
    cache_key = cfg["model_path"]
    if cache_key not in _model_cache:
        if not os.path.exists(cfg["model_path"]):
            raise FileNotFoundError(
                f"Questionnaire model not found at {cfg['model_path']}. "
                "Run: python train_all.py"
            )
        _model_cache[cache_key] = joblib.load(cfg["model_path"])
    return _model_cache[cache_key]


def get_feature_columns():
    """Load the feature column order and importance from training metadata."""
    meta_path = os.path.join(config.MODELS_DIR, "questionnaire_features.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {"feature_columns": None, "feature_importance": {}}


def predict_score(features: dict, cfg: dict = None) -> dict:
    """
    Predict mental health score from questionnaire responses.

    Parameters
    ----------
    features : dict
        Keys = feature names, values = numeric (1-10 scale).

    Returns dict with score, confidence, risk probabilities, and risk factors.
    """
    cfg = cfg or config.QUESTIONNAIRE_CONFIG
    model = load_model(cfg)
    scaler = load_scaler(cfg)

    meta = get_feature_columns()
    saved_cols = meta.get("feature_columns")
    feature_importance = meta.get("feature_importance", {})

    if saved_cols:
        X = np.array([[features.get(col, 5) for col in saved_cols]])
    else:
        X = np.array([list(features.values())])

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)

    # Weighted score: low=100, moderate=50, high=0
    weight_map = {0: 100, 1: 50, 2: 0}
    score = sum(proba[i] * weight_map.get(classes[i], 50) for i in range(len(classes)))
    score = round(max(0.0, min(100.0, score)), 2)

    confidence = round(float(max(proba)), 4)
    predicted_risk = int(model.predict(X_scaled)[0])
    risk_labels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

    risk_probs = {}
    for i in range(len(classes)):
        risk_probs[risk_labels.get(classes[i], f"Class {classes[i]}")] = round(float(proba[i]), 4)

    # Identify which answers are concerning
    concerning_items = []
    positive_items = []
    if saved_cols:
        for col in saved_cols:
            val = features.get(col, 5)
            if col == "stress_level":
                if val >= 7:
                    concerning_items.append({"item": col, "value": val, "note": "High stress"})
                elif val <= 3:
                    positive_items.append({"item": col, "value": val, "note": "Low stress"})
            else:
                if val <= 3:
                    concerning_items.append({"item": col, "value": val, "note": "Low score"})
                elif val >= 7:
                    positive_items.append({"item": col, "value": val, "note": "Good score"})

    return {
        "score": score,
        "confidence": confidence,
        "risk_probabilities": risk_probs,
        "predicted_risk": predicted_risk,
        "predicted_risk_label": risk_labels.get(predicted_risk, "Unknown"),
        "concerning_items": concerning_items,
        "positive_items": positive_items,
    }


if __name__ == "__main__":
    sample = {
        "sleep_quality": 3, "appetite": 4, "energy_level": 2,
        "social_interaction": 2, "concentration": 3, "mood": 2,
        "stress_level": 8, "physical_activity": 2,
        "interest_in_activities": 3, "self_worth": 3,
    }
    r = predict_score(sample)
    print(f"Score: {r['score']} | Confidence: {r['confidence']}")
    print(f"Risk: {r['predicted_risk_label']}")
    print(f"Concerning: {r['concerning_items']}")
