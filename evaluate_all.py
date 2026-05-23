"""
Comprehensive evaluation pipeline.

Generates:
  - Per-model comparison tables
  - Confusion matrix heatmaps (saved as PNG)
  - Feature importance charts
  - Fusion scenario tests
  - JSON metric files

Usage:
    python evaluate_all.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def evaluate_text_module():
    """Evaluate the text model on held-out test data."""
    print("=" * 60)
    print("  Evaluating TEXT Module")
    print("=" * 60)

    from src.data.text_loader import load_and_preprocess
    from src.features.text_features import load_vectorizer

    cfg = config.TEXT_CONFIG
    if not os.path.exists(cfg["model_path"]):
        print("  [SKIP] Text model not trained.")
        return

    df = load_and_preprocess(cfg)
    texts = df["text"].tolist()
    labels = df["label"].to_numpy()

    _, X_test_text, _, y_test = train_test_split(
        texts, labels,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=labels,
    )

    vectorizer = load_vectorizer(cfg)
    X_test = vectorizer.transform(X_test_text)
    model = joblib.load(cfg["model_path"])

    y_pred = model.predict(X_test)
    _print_metrics(y_test, y_pred, "text")
    _plot_confusion_matrix(y_test, y_pred, "text", ["Normal", "Distress"])


def evaluate_questionnaire_module():
    """Evaluate the questionnaire model on held-out test data."""
    print("=" * 60)
    print("  Evaluating QUESTIONNAIRE Module")
    print("=" * 60)

    from src.data.questionnaire_loader import load_and_preprocess

    cfg = config.QUESTIONNAIRE_CONFIG
    if not os.path.exists(cfg["model_path"]):
        print("  [SKIP] Questionnaire model not trained.")
        return

    df = load_and_preprocess(cfg)
    label_col = cfg["label_column"]
    feature_cols = [c for c in df.columns if c != label_col]

    X = df[feature_cols].values
    y = df[label_col].values

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    scaler = joblib.load(cfg["scaler_path"])
    X_test_scaled = scaler.transform(X_test)
    model = joblib.load(cfg["model_path"])

    y_pred = model.predict(X_test_scaled)
    _print_metrics(y_test, y_pred, "questionnaire")
    _plot_confusion_matrix(y_test, y_pred, "questionnaire", ["Low Risk", "Moderate", "High Risk"])

    # Feature importance if available
    meta_path = os.path.join(config.MODELS_DIR, "questionnaire_features.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        fi = meta.get("feature_importance", {})
        if fi:
            _plot_feature_importance(fi, "questionnaire")


def _print_metrics(y_true, y_pred, module_name: str):
    """Print and save classification metrics."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }

    print("\n  Metrics:")
    for k, v in metrics.items():
        print(f"    {k:18s}: {v}")
    print(f"\n{classification_report(y_true, y_pred, zero_division=0)}")

    eval_path = os.path.join(config.OUTPUTS_DIR, f"{module_name}_evaluation.json")
    cm = confusion_matrix(y_true, y_pred).tolist()
    with open(eval_path, "w") as f:
        json.dump({"metrics": metrics, "confusion_matrix": cm}, f, indent=2)


def _plot_confusion_matrix(y_true, y_pred, module_name: str, class_names: list):
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 14}, linewidths=0.5)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{module_name.title()} Module - Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(config.OUTPUTS_DIR, f"{module_name}_confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {path}")


def _plot_feature_importance(fi_dict: dict, module_name: str):
    """Save a horizontal bar chart of feature importance."""
    sorted_fi = sorted(fi_dict.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_fi)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ").title() for n in names])
    ax.set_xlabel("Importance")
    ax.set_title(f"{module_name.title()} - Feature Importance", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(config.OUTPUTS_DIR, f"{module_name}_feature_importance.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance plot saved to {path}")


def evaluate_fusion():
    """Run scenario tests on the fusion pipeline."""
    print("=" * 60)
    print("  Evaluating FUSION Pipeline")
    print("=" * 60)

    from src.fusion.score_fusion import fuse_scores

    scenarios = [
        {"name": "Healthy user",        "text": 88, "quest": 92, "audio": None, "t_c": 0.95, "q_c": 0.90},
        {"name": "Moderate risk",        "text": 52, "quest": 48, "audio": None, "t_c": 0.80, "q_c": 0.85},
        {"name": "High risk",           "text": 18, "quest": 22, "audio": None, "t_c": 0.93, "q_c": 0.91},
        {"name": "Mixed (text ok)",     "text": 78, "quest": 28, "audio": None, "t_c": 0.60, "q_c": 0.92},
        {"name": "Mixed (quest ok)",    "text": 25, "quest": 82, "audio": None, "t_c": 0.88, "q_c": 0.55},
        {"name": "With audio (calm)",   "text": 70, "quest": 65, "audio": 78, "t_c": 0.85, "q_c": 0.80},
        {"name": "With audio (sad)",    "text": 70, "quest": 65, "audio": 22, "t_c": 0.85, "q_c": 0.80},
    ]

    rows = []
    for s in scenarios:
        r = fuse_scores(
            text_score=s["text"], questionnaire_score=s["quest"], audio_score=s["audio"],
            text_confidence=s["t_c"], questionnaire_confidence=s["q_c"],
        )
        rows.append({
            "Scenario": s["name"],
            "Text": s["text"], "Quest": s["quest"],
            "Audio": s["audio"] or "N/A",
            "Final": r["final_score"],
            "Risk": r["risk_label"],
        })

    df = pd.DataFrame(rows)
    print(f"\n{df.to_string(index=False)}")
    df.to_csv(os.path.join(config.OUTPUTS_DIR, "fusion_scenarios.csv"), index=False)


if __name__ == "__main__":
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    evaluate_text_module()
    print()
    evaluate_questionnaire_module()
    print()
    evaluate_fusion()
    print("\n[DONE] All evaluations complete. Check outputs/")
