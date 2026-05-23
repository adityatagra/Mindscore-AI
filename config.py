"""
Central configuration for the Multimodal Mental Health Scoring System.
All paths, hyperparameters, and thresholds are defined here so nothing is hardcoded
in individual modules.
"""

import os

# ─── Project Root ────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Directory Paths ─────────────────────────────────────────────────────────
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DB_PATH = os.path.join(PROJECT_ROOT, "outputs", "user_scores.db")

# ─── Text Module Configuration ───────────────────────────────────────────────
TEXT_CONFIG = {
    "raw_csv": os.path.join(DATA_RAW_DIR, "text_dataset.csv"),
    "processed_csv": os.path.join(DATA_PROCESSED_DIR, "text_processed.csv"),
    "text_column": "text",           # column name containing the text
    "label_column": "label",         # column name containing label (1=distress, 0=normal)
    "max_features": 5000,            # TF-IDF vocabulary size
    "test_size": 0.2,
    "random_state": 42,
    "model_path": os.path.join(MODELS_DIR, "text_model.joblib"),
    "vectorizer_path": os.path.join(MODELS_DIR, "text_vectorizer.joblib"),
}

# ─── Questionnaire Module Configuration ──────────────────────────────────────
QUESTIONNAIRE_CONFIG = {
    "raw_csv": os.path.join(DATA_RAW_DIR, "questionnaire_dataset.csv"),
    "processed_csv": os.path.join(DATA_PROCESSED_DIR, "questionnaire_processed.csv"),
    "feature_columns": [],           # leave empty to auto-detect (all columns except label)
    "label_column": "risk_level",    # 0=low, 1=moderate, 2=high risk
    "test_size": 0.2,
    "random_state": 42,
    "model_path": os.path.join(MODELS_DIR, "questionnaire_model.joblib"),
    "scaler_path": os.path.join(MODELS_DIR, "questionnaire_scaler.joblib"),
}

# ─── Audio Module Configuration (Optional) ───────────────────────────────────
AUDIO_CONFIG = {
    "raw_dir": os.path.join(DATA_RAW_DIR, "audio"),
    "metadata_csv": os.path.join(DATA_RAW_DIR, "audio_metadata.csv"),
    "processed_csv": os.path.join(DATA_PROCESSED_DIR, "audio_features.csv"),
    "file_column": "filename",
    "label_column": "emotion",       # e.g. happy, sad, angry, neutral, fear
    "sample_rate": 22050,
    "n_mfcc": 13,
    "n_chroma": 12,
    "min_duration_sec": 10,          # minimum seconds required for analysis
    "max_duration_sec": 120,         # maximum seconds allowed (None = no limit)
    "test_size": 0.2,
    "random_state": 42,
    "model_path": os.path.join(MODELS_DIR, "audio_model.joblib"),
    "scaler_path": os.path.join(MODELS_DIR, "audio_scaler.joblib"),
    "enabled": True,
}

# ─── Fusion Configuration ────────────────────────────────────────────────────
FUSION_CONFIG = {
    "weights": {
        "text": 0.40,
        "questionnaire": 0.40,
        "audio": 0.20,
    },
    "score_range": (0, 100),         # final score is normalized to this range
}

# ─── Risk Band Mapping ───────────────────────────────────────────────────────
RISK_BANDS = {
    "high_risk": (0, 39),
    "moderate_risk": (40, 69),
    "stable": (70, 100),
}

# ─── Alert Thresholds ────────────────────────────────────────────────────────
ALERT_CONFIG = {
    "critical_score_threshold": 40,  # alert if score < this
    "drop_threshold": 15,            # alert if score drops by this much
    "drop_window_days": 3,           # within this many days
}

# ─── Questionnaire Items (for the Streamlit form) ────────────────────────────
QUESTIONNAIRE_ITEMS = [
    {"id": "sleep_quality", "question": "How would you rate your sleep quality over the past week?", "min": 1, "max": 10},
    {"id": "appetite", "question": "How is your appetite recently?", "min": 1, "max": 10},
    {"id": "energy_level", "question": "How would you rate your energy level?", "min": 1, "max": 10},
    {"id": "social_interaction", "question": "How often did you interact with friends/family this week?", "min": 1, "max": 10},
    {"id": "concentration", "question": "How well could you concentrate on tasks?", "min": 1, "max": 10},
    {"id": "mood", "question": "How would you describe your overall mood?", "min": 1, "max": 10},
    {"id": "stress_level", "question": "How stressed have you felt?", "min": 1, "max": 10},
    {"id": "physical_activity", "question": "How physically active have you been?", "min": 1, "max": 10},
    {"id": "interest_in_activities", "question": "How much interest do you have in your usual activities?", "min": 1, "max": 10},
    {"id": "self_worth", "question": "How do you feel about yourself / your self-worth?", "min": 1, "max": 10},
]
