"""
Questionnaire / behavioural-data loader and preprocessor.

Expects a CSV where each row is a respondent with numeric feature columns
and a label column indicating risk level.

Label encoding:
  0 = low risk  |  1 = moderate risk  |  2 = high risk

You can adapt this for any Kaggle mental-health-survey / lifestyle dataset
by updating config.QUESTIONNAIRE_CONFIG.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def load_and_preprocess(cfg: dict = None) -> pd.DataFrame:
    """
    Load raw questionnaire CSV, handle missing values, and save processed version.

    Returns
    -------
    pd.DataFrame ready for model training.
    """
    cfg = cfg or config.QUESTIONNAIRE_CONFIG
    raw_path = cfg["raw_csv"]
    processed_path = cfg["processed_csv"]

    if os.path.exists(processed_path):
        print(f"[QuestionnaireLoader] Loading cached data from {processed_path}")
        return pd.read_csv(processed_path)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Questionnaire dataset not found at {raw_path}.\n"
            "Please download a mental-health survey dataset and place it there.\n"
            f"Expected label column: '{cfg['label_column']}'."
        )

    print(f"[QuestionnaireLoader] Reading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    label_col = cfg["label_column"]
    if label_col not in df.columns:
        raise ValueError(
            f"Expected label column '{label_col}' in {raw_path}. "
            f"Found: {list(df.columns)}. Update config.QUESTIONNAIRE_CONFIG."
        )

    feature_cols = cfg["feature_columns"]
    if not feature_cols:
        feature_cols = [c for c in df.columns if c != label_col]

    df = df[feature_cols + [label_col]].copy()

    for col in feature_cols:
        if df[col].dtype == "object":
            df[col] = pd.Categorical(df[col]).codes
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df[label_col] = df[label_col].astype(int)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[QuestionnaireLoader] Saved processed data to {processed_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    data = load_and_preprocess()
    print(data.head())
    print(f"Label distribution:\n{data[config.QUESTIONNAIRE_CONFIG['label_column']].value_counts()}")
