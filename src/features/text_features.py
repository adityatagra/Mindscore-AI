"""
Text feature extraction using TF-IDF.

This module builds and persists the TF-IDF vectorizer used by the text model.
"""

import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def build_vectorizer(corpus, cfg: dict = None):
    """
    Fit a TF-IDF vectorizer on the training corpus and persist it.

    Returns
    -------
    vectorizer : fitted TfidfVectorizer
    X          : sparse TF-IDF matrix
    """
    cfg = cfg or config.TEXT_CONFIG
    vectorizer = TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(corpus)

    os.makedirs(os.path.dirname(cfg["vectorizer_path"]), exist_ok=True)
    joblib.dump(vectorizer, cfg["vectorizer_path"])
    print(f"[TextFeatures] Vectorizer saved to {cfg['vectorizer_path']}")
    return vectorizer, X


def load_vectorizer(cfg: dict = None):
    """Load a previously fitted TF-IDF vectorizer."""
    cfg = cfg or config.TEXT_CONFIG
    if not os.path.exists(cfg["vectorizer_path"]):
        raise FileNotFoundError(
            f"Vectorizer not found at {cfg['vectorizer_path']}. Train the text model first."
        )
    return joblib.load(cfg["vectorizer_path"])


def transform_text(texts, cfg: dict = None):
    """Transform raw text list into TF-IDF features using the saved vectorizer."""
    vectorizer = load_vectorizer(cfg)
    return vectorizer.transform(texts)
