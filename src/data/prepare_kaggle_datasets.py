"""
Prepare and merge multiple Kaggle datasets for BTech-scale training.

Run in Google Colab AFTER downloading datasets into data/raw/kaggle/:

    !mkdir -p data/raw/kaggle
    !kaggle datasets download -d infamouscoder/depression-reddit-cleaned -p data/raw/kaggle --unzip
    !kaggle datasets download -d suchintikasarkar/sentiment-analysis-for-mental-health -p data/raw/kaggle --unzip
    !kaggle datasets download -d shariful07/student-mental-health -p data/raw/kaggle --unzip
    !kaggle datasets download -d osmi/mental-health-in-tech-survey -p data/raw/kaggle --unzip

    %cd /content/mindscore   # your project root
    !python src/data/prepare_kaggle_datasets.py
    !rm -rf data/processed
    !python train_all.py

Outputs:
    data/raw/text_dataset.csv          (merged text, deduplicated)
    data/raw/questionnaire_dataset.csv (merged surveys)
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config

KAGGLE_DIR = os.path.join(config.DATA_RAW_DIR, "kaggle")
MAX_TEXT_ROWS = 25_000  # cap for free Colab CPU; set None for no limit
RANDOM_STATE = 42

FEATURE_COLS = [item["id"] for item in config.QUESTIONNAIRE_ITEMS]
LABEL_COL = config.QUESTIONNAIRE_CONFIG["label_column"]


def _find(pattern: str) -> str | None:
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def _yn_to_score(series: pd.Series, distress_yes: int = 8, distress_no: int = 2) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map({"yes": distress_yes, "no": distress_no})
    return mapped.fillna(5)


def _risk_from_mood(mood: float) -> int:
    if mood >= 7:
        return 0
    if mood >= 4:
        return 1
    return 2


def _clean_text_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.replace({"nan": "", "None": ""})
    return s


def prepare_reddit_text() -> pd.DataFrame | None:
    path = _find(os.path.join(KAGGLE_DIR, "**/depression_dataset_reddit_cleaned*.csv"))
    if not path:
        path = _find(os.path.join(config.DATA_RAW_DIR, "**/depression_dataset_reddit_cleaned*.csv"))
    if not path:
        print("[text] Reddit depression dataset not found — skip")
        return None

    df = pd.read_csv(path)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("clean_text", "text", "body", "post"):
            col_map[c] = "text"
        if cl in ("is_depression", "label", "class", "target"):
            col_map[c] = "label"
    df = df.rename(columns=col_map)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Reddit CSV columns not recognized at {path}: {df.columns.tolist()}")

    out = df[["text", "label"]].copy()
    out["text"] = _clean_text_series(out["text"])
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    out = out[out["text"].str.len() >= 20].dropna()
    out["source"] = "reddit_depression"
    print(f"[text] Reddit: {len(out)} rows from {os.path.basename(path)}")
    return out


def prepare_sentiment_mh_text() -> pd.DataFrame | None:
    path = _find(os.path.join(KAGGLE_DIR, "**/*.csv"))
    candidates = glob.glob(os.path.join(KAGGLE_DIR, "**/*.csv"), recursive=True)
    path = None
    for p in candidates:
        base = os.path.basename(p).lower()
        if "mental" in base and "health" in base and "sentiment" in base:
            path = p
            break
        if base in ("mental_health.csv", "combined data.csv", "combined_data.csv"):
            path = p
            break
    if not path:
        for p in candidates:
            try:
                peek = pd.read_csv(p, nrows=5)
                cols = {c.lower() for c in peek.columns}
                if ("statement" in cols or "text" in cols) and (
                    "status" in cols or "label" in cols or "class" in cols
                ):
                    path = p
                    break
            except Exception:
                continue
    if not path:
        print("[text] Sentiment Mental Health dataset not found — skip")
        return None

    df = pd.read_csv(path)
    text_col = next((c for c in df.columns if c.lower() in ("statement", "text", "post", "tweet")), None)
    label_col = next((c for c in df.columns if c.lower() in ("status", "label", "class", "category")), None)
    if not text_col or not label_col:
        print(f"[text] Sentiment MH: unrecognized columns {df.columns.tolist()} — skip")
        return None

    df = df.rename(columns={text_col: "text", label_col: "status"})
    df["text"] = _clean_text_series(df["text"])
    status = df["status"].astype(str).str.strip().str.lower()
    distress = status.isin(
        {"depression", "depressed", "suicidal", "anxiety", "stress", "bipolar", "personality disorder"}
    )
    normal = status.isin({"normal", "healthy", "none", "no issues"})
    df = df[distress | normal].copy()
    df["label"] = distress.astype(int)
    df = df[df["text"].str.len() >= 20]
    df["source"] = "sentiment_mh"
    print(f"[text] Sentiment MH: {len(df)} rows from {os.path.basename(path)}")
    return df[["text", "label", "source"]]


def merge_text_datasets() -> pd.DataFrame:
    parts = [p for p in (prepare_reddit_text(), prepare_sentiment_mh_text()) if p is not None]
    if not parts:
        raise FileNotFoundError(
            "No text datasets found. Download at least infamouscoder/depression-reddit-cleaned "
            "into data/raw/kaggle/"
        )

    merged = pd.concat(parts, ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"]).sample(frac=1, random_state=RANDOM_STATE)
    if MAX_TEXT_ROWS and len(merged) > MAX_TEXT_ROWS:
        merged = merged.sample(MAX_TEXT_ROWS, random_state=RANDOM_STATE)
        print(f"[text] Subsampled to {MAX_TEXT_ROWS} rows for Colab-friendly training")

    out = merged[["text", "label"]]
    dest = config.TEXT_CONFIG["raw_csv"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False)
    print(
        f"[text] Saved {dest} → {len(out)} rows | "
        f"label balance: {out['label'].value_counts().to_dict()}"
    )
    return out


def prepare_student_questionnaire() -> pd.DataFrame | None:
    path = _find(os.path.join(KAGGLE_DIR, "**/Student Mental health*.csv"))
    if not path:
        path = _find(os.path.join(config.DATA_RAW_DIR, "**/Student Mental health*.csv"))
    if not path:
        print("[questionnaire] Student Mental Health not found — skip")
        return None

    df = pd.read_csv(path)

    def col(*names):
        for n in names:
            for c in df.columns:
                if n.lower() in c.lower():
                    return c
        return None

    anxiety = col("anxiety")
    panic = col("panic")
    depression = col("depression")
    age = col("age")
    cgpa = col("cgpa")
    specialist = col("specialist", "treatment")

    if not all([anxiety, panic, depression]):
        print(f"[questionnaire] Student MH: missing columns {df.columns.tolist()} — skip")
        return None

    q = pd.DataFrame()
    q["sleep_quality"] = 10 - _yn_to_score(df[anxiety])
    q["appetite"] = 10 - _yn_to_score(df[panic])
    q["energy_level"] = 10 - _yn_to_score(df[depression])
    if age:
        q["social_interaction"] = pd.to_numeric(df[age], errors="coerce").fillna(20).clip(1, 10)
    else:
        q["social_interaction"] = 5
    if cgpa:
        cgpa_num = df[cgpa].astype(str).str.extract(r"([\d.]+)")[0]
        q["concentration"] = pd.to_numeric(cgpa_num, errors="coerce").fillna(5).clip(1, 10)
    else:
        q["concentration"] = 5
    q["mood"] = 10 - _yn_to_score(df[depression])
    q["stress_level"] = _yn_to_score(df[anxiety])
    q["physical_activity"] = 5
    q["interest_in_activities"] = 10 - _yn_to_score(df[depression])
    if specialist:
        q["self_worth"] = 10 - _yn_to_score(df[specialist])
    else:
        q["self_worth"] = 5
    q[LABEL_COL] = q["mood"].apply(_risk_from_mood)
    q["source"] = "student_mh"
    print(f"[questionnaire] Student MH: {len(q)} rows")
    return q


def prepare_osmi_questionnaire() -> pd.DataFrame | None:
    path = _find(os.path.join(KAGGLE_DIR, "**/survey.csv"))
    if not path:
        candidates = glob.glob(os.path.join(KAGGLE_DIR, "**/*.csv"), recursive=True)
        for p in candidates:
            if "mental-health" in p.lower() or "mental_health" in p.lower():
                path = p
                break
    if not path:
        print("[questionnaire] OSMI Mental Health in Tech not found — skip")
        return None

    df = pd.read_csv(path)

    def col(*names):
        for n in names:
            for c in df.columns:
                if n.lower() in c.lower():
                    return c
        return None

    interfere = col("mental health interfere", "work_interfere")
    treat = col("treated mental health", "treatment", "seek help")
    benefits = col("benefits")
    anonymity = col("anonymity")
    leave = col("leave")
    coworkers = col("coworkers")
    supervisors = col("supervisors")
    discuss = col("discuss")

    if not interfere:
        print(f"[questionnaire] OSMI: no work interference column in {df.columns[:8]}... — skip")
        return None

    interfere_map = {
        "never": 2, "rarely": 4, "sometimes": 6, "often": 9,
        "not sure": 5, "nan": 5,
    }
    interfere_score = (
        df[interfere].astype(str).str.strip().str.lower().map(interfere_map).fillna(5)
    )

    q = pd.DataFrame()
    q["sleep_quality"] = 10 - interfere_score.clip(1, 9)
    q["appetite"] = 10 - interfere_score.clip(1, 9)
    q["energy_level"] = 10 - interfere_score.clip(1, 9)
    q["social_interaction"] = 5
    if discuss:
        discuss_map = {"yes": 8, "no": 3, "not sure": 5, "maybe": 5}
        q["concentration"] = df[discuss].astype(str).str.lower().map(discuss_map).fillna(5)
    else:
        q["concentration"] = 5
    q["mood"] = 10 - interfere_score.clip(1, 9)
    q["stress_level"] = interfere_score.clip(1, 10)
    q["physical_activity"] = 5
    q["interest_in_activities"] = 10 - interfere_score.clip(1, 9)
    if treat:
        q["self_worth"] = _yn_to_score(df[treat], distress_yes=3, distress_no=8)
    elif benefits:
        q["self_worth"] = _yn_to_score(df[benefits], distress_yes=4, distress_no=7)
    else:
        q["self_worth"] = 5
    q[LABEL_COL] = q["mood"].apply(_risk_from_mood)
    q["source"] = "osmi_tech"
    print(f"[questionnaire] OSMI Tech: {len(q)} rows from {os.path.basename(path)}")
    return q


def merge_questionnaire_datasets() -> pd.DataFrame:
    parts = [p for p in (prepare_student_questionnaire(), prepare_osmi_questionnaire()) if p is not None]
    if not parts:
        raise FileNotFoundError(
            "No questionnaire datasets found. Download student-mental-health and/or "
            "osmi/mental-health-in-tech-survey into data/raw/kaggle/"
        )

    merged = pd.concat(parts, ignore_index=True)
    merged = merged[FEATURE_COLS + [LABEL_COL]].dropna()
    merged[FEATURE_COLS] = merged[FEATURE_COLS].clip(1, 10)

    dest = config.QUESTIONNAIRE_CONFIG["raw_csv"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    merged.to_csv(dest, index=False)
    print(
        f"[questionnaire] Saved {dest} → {len(merged)} rows | "
        f"risk: {merged[LABEL_COL].value_counts().to_dict()}"
    )
    return merged


def main():
    print("=" * 60)
    print("  Kaggle dataset preparation (BTech multi-source merge)")
    print("=" * 60)
    print(f"  Looking in: {KAGGLE_DIR}\n")

    merge_text_datasets()
    print()
    merge_questionnaire_datasets()
    print("\nDone. Next: rm -rf data/processed && python train_all.py")


if __name__ == "__main__":
    main()
