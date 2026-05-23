"""
Advanced questionnaire model training pipeline.

Strategy:
  1. Scale features with StandardScaler
  2. Train & compare multiple classifiers:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - Extra Trees
  3. Hyperparameter tuning via GridSearchCV
  4. Build a stacked ensemble
  5. Feature importance analysis
  6. Full evaluation suite
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
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.data.questionnaire_loader import load_and_preprocess
from src.features.questionnaire_features import build_scaler

CV_FOLDS = 5
RANDOM_STATE = 42


def _evaluate(name, model, X_test, y_test):
    """Evaluate a single model."""
    y_pred = model.predict(X_test)
    m = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }
    return m, y_pred


def train(cfg: dict = None):
    """Full advanced training pipeline for the questionnaire module."""
    cfg = cfg or config.QUESTIONNAIRE_CONFIG
    print("=" * 65)
    print("  QUESTIONNAIRE MODULE - Advanced Training Pipeline")
    print("  (Multi-model Comparison + Tuning + Ensemble)")
    print("=" * 65)

    # 1. Load data
    df = load_and_preprocess(cfg)
    label_col = cfg["label_column"]
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values
    y = df[label_col].values
    print(f"Dataset: {len(df)} samples | Features: {len(feature_cols)} | Classes: {np.unique(y)}")

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Scale
    _, X_train_scaled = build_scaler(X_train, cfg)
    from src.features.questionnaire_features import load_scaler
    scaler = load_scaler(cfg)
    X_test_scaled = scaler.transform(X_test)

    # 4. Candidate models
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=15, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=400, max_depth=15, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    # 5. Cross-validate
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    print(f"\n{'Model':<22} {'CV F1 (mean)':>12} {'CV F1 (std)':>12} {'Test F1':>10}")
    print("-" * 60)

    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        metrics, _ = _evaluate(name, model, X_test_scaled, y_test)

        results[name] = {
            "model": model,
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "test_metrics": metrics,
        }
        print(f"{name:<22} {cv_scores.mean():>12.4f} {cv_scores.std():>12.4f} {metrics['f1_score']:>10.4f}")

    # 6. Hyperparameter tuning for Gradient Boosting
    print("\n--- Hyperparameter Tuning (Gradient Boosting) ---")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    }
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="f1_weighted", n_jobs=-1, refit=True,
    )
    gs.fit(X_train_scaled, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV F1:  {gs.best_score_:.4f}")
    tuned_gb = gs.best_estimator_
    tuned_metrics, _ = _evaluate("TunedGB", tuned_gb, X_test_scaled, y_test)
    print(f"  Test F1:     {tuned_metrics['f1_score']:.4f}")

    # 7. Build Voting Ensemble
    print("\n--- Building Voting Ensemble ---")
    sorted_models = sorted(results.items(), key=lambda x: x[1]["cv_f1_mean"], reverse=True)
    ensemble_estimators = [(name, res["model"]) for name, res in sorted_models[:3]]
    ensemble_estimators.append(("TunedGB", tuned_gb))
    print(f"  Components: {[n for n, _ in ensemble_estimators]}")

    ensemble = VotingClassifier(
        estimators=ensemble_estimators,
        voting="soft",
        n_jobs=-1,
    )
    ensemble.fit(X_train_scaled, y_train)

    calibrated = CalibratedClassifierCV(ensemble, cv=3, method="isotonic")
    calibrated.fit(X_train_scaled, y_train)

    ens_metrics, y_pred_ens = _evaluate("CalibratedEnsemble", calibrated, X_test_scaled, y_test)
    cm = confusion_matrix(y_test, y_pred_ens).tolist()
    report = classification_report(y_test, y_pred_ens, zero_division=0)

    print(f"\n--- FINAL MODEL: Calibrated Voting Ensemble ---")
    for k, v in ens_metrics.items():
        print(f"  {k:12s}: {v}")
    print(f"\nConfusion Matrix:\n{np.array(cm)}")
    print(f"\nClassification Report:\n{report}")

    # 8. Feature importance (find a tree-based model that has it)
    fi_dict = {}
    # First try tuned GB, then check candidates in order
    fi_sources = [("TunedGB", tuned_gb)] + [(n, r["model"]) for n, r in sorted_models]
    for src_name, src_model in fi_sources:
        if hasattr(src_model, "feature_importances_"):
            importances = src_model.feature_importances_
            fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            print(f"\n--- Feature Importance (from {src_name}) ---")
            for fname, imp in fi:
                bar = "#" * int(imp * 80)
                print(f"  {fname:<25s}: {imp:.4f} {bar}")
            fi_dict = {fname: round(float(imp), 4) for fname, imp in fi}
            break
    if not fi_dict:
        print("\n[INFO] No tree-based model found for feature importance.")

    # 9. Save
    os.makedirs(os.path.dirname(cfg["model_path"]), exist_ok=True)
    joblib.dump(calibrated, cfg["model_path"])
    print(f"\nModel saved to {cfg['model_path']}")

    meta_path = os.path.join(config.MODELS_DIR, "questionnaire_features.json")
    with open(meta_path, "w") as f:
        json.dump({"feature_columns": feature_cols, "feature_importance": fi_dict}, f, indent=2)

    all_results = {}
    for name, res in results.items():
        all_results[name] = {
            "cv_f1_mean": res["cv_f1_mean"],
            "cv_f1_std": res["cv_f1_std"],
            "test_metrics": res["test_metrics"],
        }
    all_results["TunedGradientBoosting"] = {
        "best_params": {k: (int(v) if isinstance(v, np.integer) else v) for k, v in gs.best_params_.items()},
        "best_cv_f1": round(gs.best_score_, 4),
        "test_metrics": tuned_metrics,
    }
    all_results["CalibratedEnsemble"] = {
        "components": [n for n, _ in ensemble_estimators],
        "test_metrics": ens_metrics,
        "confusion_matrix": cm,
    }
    all_results["feature_importance"] = fi_dict

    eval_path = os.path.join(config.OUTPUTS_DIR, "questionnaire_evaluation.json")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(eval_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full evaluation saved to {eval_path}")

    return calibrated, ens_metrics


if __name__ == "__main__":
    train()
