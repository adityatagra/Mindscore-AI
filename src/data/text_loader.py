"""
Text dataset loader and preprocessor.

Expects a CSV with at least two columns:
  - text column  : raw text (diary entry, social media post, etc.)
  - label column : binary label  (1 = distress/depression, 0 = normal)

You can plug in any Kaggle text-depression dataset by updating config.TEXT_CONFIG.
"""

import os
import re

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# Built-in stopwords so we never depend on an NLTK download
_STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "her", "hers", "herself", "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
    "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
    "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
    "shouldn", "wasn", "weren", "won", "wouldn",
}

def _lemmatize(word: str) -> str:
    """Simple suffix-based stemming that needs no external data downloads."""
    for suffix in ("ness", "ment", "tion", "sion", "ing", "ful", "less", "able", "ible", "ly", "ed", "er", "est"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def clean_text(text: str) -> str:
    """Lower-case, strip URLs/mentions/punctuation, lemmatize, remove stopwords."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [_lemmatize(t) for t in tokens if t not in _STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def load_and_preprocess(cfg: dict = None) -> pd.DataFrame:
    """
    Load raw CSV, clean text, and save processed version.

    Returns
    -------
    pd.DataFrame with columns ['text', 'label']
    """
    cfg = cfg or config.TEXT_CONFIG
    raw_path = cfg["raw_csv"]
    processed_path = cfg["processed_csv"]

    if os.path.exists(processed_path):
        print(f"[TextLoader] Loading cached processed data from {processed_path}")
        return pd.read_csv(processed_path)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Text dataset not found at {raw_path}.\n"
            "Please download a depression/stress text dataset from Kaggle and place it there.\n"
            f"Expected columns: '{cfg['text_column']}' and '{cfg['label_column']}'."
        )

    print(f"[TextLoader] Reading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    text_col = cfg["text_column"]
    label_col = cfg["label_column"]
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' in {raw_path}. "
            f"Found: {list(df.columns)}. Update config.TEXT_CONFIG accordingly."
        )

    df = df[[text_col, label_col]].dropna().copy()
    df.rename(columns={text_col: "text", label_col: "label"}, inplace=True)
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.strip().astype(bool)]

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"[TextLoader] Saved processed data to {processed_path} ({len(df)} samples)")
    return df


if __name__ == "__main__":
    data = load_and_preprocess()
    print(data.head())
    print(f"Label distribution:\n{data['label'].value_counts()}")
