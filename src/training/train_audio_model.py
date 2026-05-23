"""
Advanced audio emotion classifier training.

Pipeline:
  1. Load 34-dimensional audio features (MFCC + spectral)
  2. Scale with StandardScaler
  3. Train & compare: SVM, Random Forest, Gradient Boosting
  4. Build calibrated ensemble
  5. Save model, label encoder, and evaluation
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.features.audio_features import build_scaler

RANDOM_STATE = 42


def train(cfg: dict = None):
    """Full training pipeline for the audio module."""
    cfg = cfg or config.AUDIO_CONFIG

    if not cfg["enabled"]:
        print("[AudioTrainer] Audio module is disabled. Skipping.")
        return None, None

    print("=" * 65)
    print("  AUDIO MODULE - Advanced Training Pipeline")
    print("  (Multi-model + Calibrated Ensemble)")
    print("=" * 65)

    # 1. Load features
    processed_path = cfg["processed_csv"]
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Audio features not found at {processed_path}. "
            "Run generate_dummy_data.py or provide audio data first."
        )

    df = pd.read_csv(processed_path)
    label_col = cfg["label_column"]
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values
    y_raw = df[label_col].values

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    le_path = os.path.join(config.MODELS_DIR, "audio_label_encoder.joblib")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    joblib.dump(le, le_path)
    print(f"Classes: {list(le.classes_)}")
    print(f"Dataset: {len(df)} samples | Features: {len(feature_cols)}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=RANDOM_STATE, stratify=y,
    )

    # 3. Scale
    _, X_train_scaled = build_scaler(X_train, cfg)
    from src.features.audio_features import load_scaler
    scaler = load_scaler(cfg)
    X_test_scaled = scaler.transform(X_test)

    # 4. Candidate models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = {
        "SVM_RBF": SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                       class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE),
    }

    results = {}
    print(f"\n{'Model':<22} {'CV F1':>10} {'Test F1':>10} {'Test Acc':>10}")
    print("-" * 55)

    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                                    scoring="f1_weighted", n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        test_acc = accuracy_score(y_test, y_pred)

        results[name] = {"model": model, "cv_f1": cv_scores.mean(), "test_f1": test_f1}
        print(f"{name:<22} {cv_scores.mean():>10.4f} {test_f1:>10.4f} {test_acc:>10.4f}")

    # 5. Build ensemble
    print("\n--- Building Calibrated Ensemble ---")
    sorted_models = sorted(results.items(), key=lambda x: x[1]["cv_f1"], reverse=True)
    ensemble_estimators = [(n, r["model"]) for n, r in sorted_models]

    ensemble = VotingClassifier(estimators=ensemble_estimators, voting="soft", n_jobs=-1)
    ensemble.fit(X_train_scaled, y_train)

    calibrated = CalibratedClassifierCV(ensemble, cv=3, method="isotonic")
    calibrated.fit(X_train_scaled, y_train)

    y_pred = calibrated.predict(X_test_scaled)
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print(f"\n--- FINAL: Calibrated Audio Ensemble ---")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\nConfusion Matrix:\n{np.array(cm)}")
    print(f"\n{report}")

    # 6. Save
    joblib.dump(calibrated, cfg["model_path"])
    print(f"Model saved to {cfg['model_path']}")

    eval_path = os.path.join(config.OUTPUTS_DIR, "audio_evaluation.json")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm,
                    "classes": list(le.classes_)}, f, indent=2)

    return calibrated, metrics


if __name__ == "__main__":
    train()
