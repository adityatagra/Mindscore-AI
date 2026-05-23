"""
Audio feature scaler (optional module).

Standardizes MFCC features before training the audio emotion classifier.
"""

import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build_scaler(X, cfg: dict = None):
    """Fit and persist a StandardScaler for audio features."""
    cfg = cfg or config.AUDIO_CONFIG
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs(os.path.dirname(cfg["scaler_path"]), exist_ok=True)
    joblib.dump(scaler, cfg["scaler_path"])
    print(f"[AudioFeatures] Scaler saved to {cfg['scaler_path']}")
    return scaler, X_scaled


def load_scaler(cfg: dict = None):
    """Load a previously fitted audio scaler."""
    cfg = cfg or config.AUDIO_CONFIG
    if not os.path.exists(cfg["scaler_path"]):
        raise FileNotFoundError(
            f"Audio scaler not found at {cfg['scaler_path']}. Train the audio model first."
        )
    return joblib.load(cfg["scaler_path"])


def transform_features(X, cfg: dict = None):
    """Scale MFCC features using the saved scaler."""
    scaler = load_scaler(cfg)
    return scaler.transform(X)
