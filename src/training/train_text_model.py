"""
Advanced text model training pipeline.

Strategy:
  1. TF-IDF feature extraction with optimized parameters
  2. Train & compare multiple classifiers:
     - Logistic Regression (baseline)
     - Linear SVM
     - Random Forest
     - Gradient Boosting
  3. Hyperparameter tuning via GridSearchCV with 5-fold stratified CV
  4. Build a Voting Ensemble of top performers
  5. Calibrate probabilities with CalibratedClassifierCV
  6. Full evaluation with metrics, confusion matrix, and feature importance
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.text_loader import load_and_preprocess
from src.features.text_features import build_vectorizer

CV_FOLDS = 5
RANDOM_STATE = 42


def _evaluate(name, model, X_test, y_test):
    """Evaluate a single model and return metrics dict."""
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = round(roc_auc_score(y_test, y_proba), 4)
    except Exception:
        auc = None

    m = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    if auc is not None:
        m["roc_auc"] = auc
    return m, y_pred


def train(cfg: dict = None):
    """Full advanced training pipeline for the text module."""
    cfg = cfg or config.TEXT_CONFIG
    print("=" * 65)
    print("  TEXT MODULE - Advanced Training Pipeline")
    print("  (Ensemble + Tuning + Calibration)")
    print("=" * 65)

    # 1. Load data
    df = load_and_preprocess(cfg)
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()
    print(f"Dataset: {len(df)} samples | Distress: {(labels==1).sum()} | Normal: {(labels==0).sum()}")

    # 2. Split
    X_text_train, X_text_test, y_train, y_test = train_test_split(
        texts, labels,
        test_size=cfg["test_size"],
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # 3. Build TF-IDF
    vectorizer, X_train = build_vectorizer(X_text_train, cfg)
    X_test = vectorizer.transform(X_text_test)
    print(f"TF-IDF: {X_train.shape[1]} features\n")

    # 4. Define candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, C=1.0,
        ),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, C=1.0),
            cv=3,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }

    # 5. Cross-validate each
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    print(f"{'Model':<22} {'CV F1 (mean)':>12} {'CV F1 (std)':>12} {'Test F1':>10}")
    print("-" * 60)

    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        model.fit(X_train, y_train)
        metrics, y_pred = _evaluate(name, model, X_test, y_test)

        results[name] = {
            "model": model,
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "test_metrics": metrics,
        }
        print(f"{name:<22} {cv_scores.mean():>12.4f} {cv_scores.std():>12.4f} {metrics['f1_score']:>10.4f}")

    # 6. Hyperparameter tuning for Logistic Regression (the one most likely to benefit)
    print("\n--- Hyperparameter Tuning (Logistic Regression) ---")
    param_grid = {
        "C": [0.01, 0.1, 1.0, 5.0, 10.0],
        "penalty": ["l2"],
    }
    gs = GridSearchCV(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV F1:  {gs.best_score_:.4f}")
    tuned_lr = gs.best_estimator_
    tuned_metrics, _ = _evaluate("TunedLR", tuned_lr, X_test, y_test)
    print(f"  Test F1:     {tuned_metrics['f1_score']:.4f}")

    # 7. Build Voting Ensemble from top 3 models
    print("\n--- Building Voting Ensemble ---")
    sorted_models = sorted(results.items(), key=lambda x: x[1]["cv_f1_mean"], reverse=True)
    top_models = sorted_models[:3]
    ensemble_estimators = [(name, res["model"]) for name, res in top_models]

    # Add tuned LR
    ensemble_estimators.append(("TunedLR", tuned_lr))
    print(f"  Components: {[n for n, _ in ensemble_estimators]}")

    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train, y_train)

    # Calibrate the ensemble for well-calibrated probabilities
    calibrated_ensemble = CalibratedClassifierCV(ensemble, cv=3, method="isotonic")
    calibrated_ensemble.fit(X_train, y_train)

    ensemble_metrics, y_pred_ensemble = _evaluate("CalibratedEnsemble", calibrated_ensemble, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred_ensemble).tolist()
    report = classification_report(y_test, y_pred_ensemble, zero_division=0)

    print(f"\n--- FINAL MODEL: Calibrated Voting Ensemble ---")
    for k, v in ensemble_metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\nConfusion Matrix:\n{np.array(cm)}")
    print(f"\nClassification Report:\n{report}")

    # 8. Save the best model
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    joblib.dump(calibrated_ensemble, cfg["model_path"])
    print(f"Model saved to {cfg['model_path']}")

    # 9. Save detailed evaluation
    all_results = {}
    for name, res in results.items():
        all_results[name] = {
            "cv_f1_mean": res["cv_f1_mean"],
            "cv_f1_std": res["cv_f1_std"],
            "test_metrics": res["test_metrics"],
        }
    all_results["TunedLogisticRegression"] = {
        "best_params": gs.best_params_,
        "best_cv_f1": round(gs.best_score_, 4),
        "test_metrics": tuned_metrics,
    }
    all_results["CalibratedEnsemble"] = {
        "components": [n for n, _ in ensemble_estimators],
        "test_metrics": ensemble_metrics,
        "confusion_matrix": cm,
    }

    eval_path = os.path.join(config.OUTPUTS_DIR, "text_evaluation.json")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full evaluation saved to {eval_path}")

    return calibrated_ensemble, ensemble_metrics


if __name__ == "__main__":
    train()
