"""
Advanced Trend Tracking Module.

Stores user mental health scores in SQLite with timestamps and provides:
  - Score history retrieval
  - Moving average computation
  - Trend direction detection (improving / stable / declining)
  - Statistical summary
"""

import os
import sys
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def _get_connection() -> sqlite3.Connection:
    """Open the database and ensure schema exists."""
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             TEXT    NOT NULL DEFAULT 'default_user',
            timestamp           TEXT    NOT NULL,
            final_score         REAL    NOT NULL,
            text_score          REAL,
            questionnaire_score REAL,
            audio_score         REAL,
            risk_band           TEXT,
            text_confidence     REAL,
            quest_confidence    REAL,
            audio_confidence    REAL
        )
    """)
    conn.commit()
    return conn


def save_score(
    final_score: float,
    text_score: float = None,
    questionnaire_score: float = None,
    audio_score: float = None,
    risk_band: str = None,
    text_confidence: float = None,
    quest_confidence: float = None,
    audio_confidence: float = None,
    user_id: str = "default_user",
    timestamp: str = None,
):
    """Insert a new score record."""
    timestamp = timestamp or datetime.now().isoformat()
    conn = _get_connection()
    conn.execute(
        """INSERT INTO scores
           (user_id, timestamp, final_score, text_score, questionnaire_score,
            audio_score, risk_band, text_confidence, quest_confidence, audio_confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, timestamp, final_score, text_score, questionnaire_score,
         audio_score, risk_band, text_confidence, quest_confidence, audio_confidence),
    )
    conn.commit()
    conn.close()


def get_history(user_id: str = "default_user", limit: int = 200) -> pd.DataFrame:
    """Retrieve score history, most recent first."""
    conn = _get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM scores WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        conn, params=(user_id, limit),
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_recent_scores(user_id: str = "default_user", days: int = 7) -> pd.DataFrame:
    """Get scores from the last N days."""
    conn = _get_connection()
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    df = pd.read_sql_query(
        "SELECT * FROM scores WHERE user_id = ? AND timestamp >= ? ORDER BY timestamp ASC",
        conn, params=(user_id, cutoff),
    )
    conn.close()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def compute_moving_average(history: pd.DataFrame, window: int = 3) -> pd.Series:
    """Compute a rolling moving average on the final_score column."""
    if history.empty or len(history) < 2:
        return pd.Series(dtype=float)
    sorted_h = history.sort_values("timestamp")
    return sorted_h["final_score"].rolling(window=min(window, len(sorted_h)), min_periods=1).mean()


def detect_trend(history: pd.DataFrame, window: int = 5) -> dict:
    """
    Detect the trend direction from recent scores.

    Uses linear regression slope over the last `window` records.

    Returns dict with trend direction, slope, and interpretation.
    """
    if history.empty or len(history) < 2:
        return {"direction": "insufficient_data", "slope": 0.0, "interpretation": "Not enough data to determine trend."}

    sorted_h = history.sort_values("timestamp").tail(window)
    scores = sorted_h["final_score"].values
    x = np.arange(len(scores), dtype=float)

    if len(scores) < 2:
        return {"direction": "insufficient_data", "slope": 0.0, "interpretation": "Not enough data."}

    slope = float(np.polyfit(x, scores, 1)[0])

    if slope > 2.0:
        direction = "improving"
        interp = f"Scores are trending upward (slope: +{slope:.1f} per assessment). Good progress."
    elif slope < -2.0:
        direction = "declining"
        interp = f"Scores are trending downward (slope: {slope:.1f} per assessment). Monitor closely."
    else:
        direction = "stable"
        interp = f"Scores are relatively stable (slope: {slope:+.1f} per assessment)."

    return {"direction": direction, "slope": round(slope, 2), "interpretation": interp}


def get_statistics(user_id: str = "default_user") -> dict:
    """Compute summary statistics from all history."""
    history = get_history(user_id=user_id, limit=1000)
    if history.empty:
        return {}

    scores = history["final_score"]
    return {
        "total_assessments": len(history),
        "latest_score": round(float(scores.iloc[0]), 2),
        "mean_score": round(float(scores.mean()), 2),
        "median_score": round(float(scores.median()), 2),
        "std_score": round(float(scores.std()), 2),
        "min_score": round(float(scores.min()), 2),
        "max_score": round(float(scores.max()), 2),
        "first_assessment": str(history["timestamp"].min()),
        "last_assessment": str(history["timestamp"].max()),
    }


def clear_history(user_id: str = "default_user"):
    """Delete all scores for a user."""
    conn = _get_connection()
    conn.execute("DELETE FROM scores WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
