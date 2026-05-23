"""
Questionnaire feature scaler.

Standardizes numeric features using StandardScaler before training the
tabular ML model. The fitted scaler is persisted for use during inference.
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build_scaler(X, cfg: dict = None):
    """
    Fit a StandardScaler and persist it.

    Returns
    -------
    scaler   : fitted StandardScaler
    X_scaled : np.ndarray
    """
    cfg = cfg or config.QUESTIONNAIRE_CONFIG
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(cfg["scaler_path"]), exist_ok=True)
    joblib.dump(scaler, cfg["scaler_path"])
    print(f"[QuestionnaireFeatures] Scaler saved to {cfg['scaler_path']}")
    return scaler, X_scaled


def load_scaler(cfg: dict = None):
    """Load a previously fitted scaler."""
    cfg = cfg or config.QUESTIONNAIRE_CONFIG
    if not os.path.exists(cfg["scaler_path"]):
        raise FileNotFoundError(
            f"Scaler not found at {cfg['scaler_path']}. Train the questionnaire model first."
        )
    return joblib.load(cfg["scaler_path"])


def transform_features(X, cfg: dict = None):
    """Scale raw feature array using the saved scaler."""
    scaler = load_scaler(cfg)
    return scaler.transform(X)
