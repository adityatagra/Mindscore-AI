---
title: MindScore AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.40.0"
app_file: app/streamlit_app.py
pinned: false
license: mit
short_description: Multimodal mental health wellness scoring (text, questionnaire, audio)
---

# MindScore AI - Multimodal Mental Health Scoring & Early Warning System

A **continuous-score, multimodal AI system** that predicts mental health wellness on a 0-100 scale using ensemble machine learning, text analysis, behavioral questionnaires, and optional voice emotion recognition. The system tracks scores over time, detects worsening trends, and triggers multi-level alerts.

> **Final-Year BTech Project** — production-grade architecture, advanced ML pipeline, and professional UI.

---

## Table of Contents

1. [Project Abstract](#project-abstract)
2. [Architecture Overview](#architecture-overview)
3. [Key Technical Features](#key-technical-features)
4. [Folder Structure](#folder-structure)
5. [Setup & Installation](#setup--installation)
6. [How to Run](#how-to-run)
7. [Dataset Instructions](#dataset-instructions)
8. [Scoring Pipeline](#scoring-pipeline)
9. [Model Performance](#model-performance)
10. [Alert System](#alert-system)
11. [Evaluation](#evaluation)
12. [Google Colab Guide](#google-colab-guide)
13. [Implementation Methodology](#implementation-methodology)
14. [Limitations](#limitations)
15. [Future Enhancements](#future-enhancements)
16. [Viva Q&A Guide](#viva-qa-guide)

---

## Project Abstract

Mental health assessment in AI systems has typically relied on simplistic categorical classification (e.g., "depressed" vs. "not depressed"). This project advances beyond that paradigm by introducing a **continuous scoring framework (0-100)** that fuses outputs from multiple independent modalities using **calibrated ensemble classifiers**. The text module employs a TF-IDF feature pipeline feeding a voting ensemble of Logistic Regression, Linear SVM, Random Forest, and Gradient Boosting with isotonic calibration. The questionnaire module uses a parallel ensemble architecture with hyperparameter-tuned Gradient Boosting. A **confidence-weighted fusion** module combines modality-level scores, weighting more certain predictions higher. A trend-tracking subsystem with moving averages and linear regression detects improving, stable, or declining trajectories. A multi-level alert engine monitors for critical thresholds, rapid declines, sustained deterioration, and module-specific red flags. The system is demonstrated through a professional Streamlit dashboard with gauge charts, radar visualizations, downloadable reports, and historical analytics.

---

## Architecture Overview

```
                         MULTIMODAL MENTAL HEALTH SCORING SYSTEM
  +------------------+    +-------------------+    +------------------+
  |   TEXT MODULE     |    |  QUESTIONNAIRE    |    |  AUDIO MODULE    |
  |                  |    |  MODULE           |    |  (Optional)      |
  | TF-IDF (5000)    |    |  StandardScaler   |    |  MFCC (13)       |
  | Voting Ensemble: |    |  Voting Ensemble: |    |  SVM (RBF)       |
  |  - Logistic Reg  |    |  - Logistic Reg   |    |  + Calibration   |
  |  - Linear SVM    |    |  - Random Forest  |    |                  |
  |  - Random Forest |    |  - Gradient Boost |    |                  |
  |  - Tuned LR (GS) |    |  - Extra Trees    |    |                  |
  | + Calibration    |    |  - Tuned GB (GS)  |    |                  |
  |                  |    |  + Calibration    |    |                  |
  | Output: Score    |    |  Output: Score    |    |  Output: Score   |
  |       + Conf     |    |        + Conf     |    |        + Conf    |
  +--------+---------+    +---------+---------+    +--------+---------+
           |                        |                       |
           +------------------------+-----------------------+
                                    |
                    +---------------v-----------------+
                    |    CONFIDENCE-WEIGHTED FUSION    |
                    |  w_eff = w_base * confidence     |
                    |  score = Sum(w_i * score_i)      |
                    |  + Risk Factor Analysis          |
                    +---------------+-----------------+
                                    |
                    +---------------v-----------------+
                    |       FINAL SCORE (0-100)        |
                    |  + Risk Band + Explanation       |
                    +---+------------------+----------+
                        |                  |
            +-----------v------+  +--------v-----------+
            | TREND TRACKER    |  | ALERT SYSTEM       |
            | - SQLite storage |  | - Critical score   |
            | - Moving average |  | - Sharp drop       |
            | - Slope detect   |  | - Sustained decline|
            | - Statistics     |  | - Module-specific  |
            +------------------+  +--------------------+
```

---

## Key Technical Features

| Feature | Details |
|---------|---------|
| **Ensemble Models** | Voting classifier combining 4 algorithms per module with soft voting |
| **Hyperparameter Tuning** | GridSearchCV with 5-fold stratified cross-validation |
| **Probability Calibration** | Isotonic regression ensures well-calibrated confidence estimates |
| **Confidence-Weighted Fusion** | Modules with higher certainty get proportionally more influence |
| **Trend Detection** | Linear regression slope analysis over recent assessments |
| **Moving Averages** | 3-point rolling average smooths noise in trend visualization |
| **Multi-Level Alerts** | Critical, high, medium severity with actionable recommendations |
| **Risk Factor Analysis** | Identifies which modules drive the score up or down |
| **Professional UI** | Gauge charts, radar plots, downloadable reports, analytics dashboard |
| **Model Caching** | Inference modules cache loaded models for sub-second response |

---

## Folder Structure

```
project/
├── config.py                           # Central configuration
├── train_all.py                        # Master training pipeline
├── evaluate_all.py                     # Full evaluation suite
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                            # Place datasets here
│   │   ├── text_dataset.csv            # 1456 samples (auto-generated)
│   │   └── questionnaire_dataset.csv   # 2000 samples (auto-generated)
│   └── processed/                      # Auto-generated cleaned data
│
├── models/                             # Saved .joblib model files
│   ├── text_model.joblib
│   ├── text_vectorizer.joblib
│   ├── questionnaire_model.joblib
│   ├── questionnaire_scaler.joblib
│   └── questionnaire_features.json
│
├── src/
│   ├── data/
│   │   ├── text_loader.py              # Text preprocessing (no NLTK dependency)
│   │   ├── questionnaire_loader.py     # Tabular preprocessing
│   │   ├── audio_loader.py             # MFCC extraction (optional)
│   │   └── generate_dummy_data.py      # 1500+ sample synthetic generator
│   ├── features/
│   │   ├── text_features.py            # TF-IDF vectorizer pipeline
│   │   ├── questionnaire_features.py   # StandardScaler pipeline
│   │   └── audio_features.py           # Audio scaler
│   ├── training/
│   │   ├── train_text_model.py         # Ensemble + tuning + calibration
│   │   ├── train_questionnaire_model.py
│   │   └── train_audio_model.py
│   ├── inference/
│   │   ├── text_inference.py           # Score + confidence + quality
│   │   ├── questionnaire_inference.py  # Score + risk factors
│   │   └── audio_inference.py
│   ├── fusion/
│   │   └── score_fusion.py             # Confidence-weighted + explainability
│   └── utils/
│       ├── trend_tracker.py            # SQLite + moving avg + trend detect
│       ├── alert_system.py             # 4-level alert engine
│       └── scoring.py                  # Report generation + interpretation
│
├── app/
│   └── streamlit_app.py                # Professional 4-page dashboard
│
├── outputs/                            # Evaluation reports & plots
│   ├── text_evaluation.json
│   ├── text_confusion_matrix.png
│   ├── questionnaire_evaluation.json
│   ├── questionnaire_confusion_matrix.png
│   ├── questionnaire_feature_importance.png
│   └── fusion_scenarios.csv
│
└── notebooks/
    └── exploration.ipynb               # Data exploration notebook
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Navigate to the project
cd "depression detection using text and speech"

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Quick Start (with synthetic data)

```bash
# Step 1: Generate comprehensive synthetic datasets
python src/data/generate_dummy_data.py

# Step 2: Train all ensemble models (takes 5-10 minutes)
python train_all.py

# Step 3: Run evaluation and generate plots
python evaluate_all.py

# Step 4: Launch the dashboard
streamlit run app/streamlit_app.py
```

### With Real Kaggle Datasets

1. Download your chosen dataset from Kaggle
2. Place the CSV in `data/raw/`
3. Update column names in `config.py` if they differ
4. Delete `data/processed/` to force re-preprocessing
5. Run `python train_all.py`

---

## Dataset Instructions

### Text Dataset

Place a CSV at `data/raw/text_dataset.csv` with columns:
- `text` — raw text content (diary entry, post, message)
- `label` — binary label (1 = distress, 0 = normal)

**Recommended Kaggle datasets:**
- [Depression Reddit Cleaned](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned) — 7731 posts
- [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- [Suicide Watch (Reddit)](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) — 232K posts

### Questionnaire Dataset

Place a CSV at `data/raw/questionnaire_dataset.csv` with:
- Numeric feature columns (matching the 10 items in `config.QUESTIONNAIRE_ITEMS`)
- Label column `risk_level` (0=low, 1=moderate, 2=high risk)

**Recommended:**
- [Student Mental Health](https://www.kaggle.com/datasets/shariful07/student-mental-health)
- [Mental Health in Tech](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

### Audio Dataset (Optional)

- Place `.wav` files in `data/raw/audio/`
- Create `data/raw/audio_metadata.csv` with: `filename`, `emotion`
- Set `AUDIO_CONFIG["enabled"] = True` in `config.py`

**Recommended:**
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

---

## Scoring Pipeline

```
Text Input --> P(distress) via calibrated ensemble
                --> text_score = (1 - P(distress)) * 100
                --> text_confidence = max(P(class_i))

Questionnaire --> P(low, moderate, high) via calibrated ensemble
                --> q_score = P(low)*100 + P(mod)*50 + P(high)*0
                --> q_confidence = max(P(class_i))

Audio (opt) --> P(emotions) via calibrated SVM
              --> audio_score = Sum(P(emotion) * health_value)

FUSION:
  effective_weight[i] = base_weight[i] * confidence[i]
  normalized_weight[i] = effective_weight[i] / Sum(effective_weights)
  final_score = Sum(normalized_weight[i] * module_score[i])

  Base weights: text=0.40, questionnaire=0.40, audio=0.20
  (auto re-normalized when a module is missing)

RISK BANDS:
   0-39 = High Risk      (Red)
  40-69 = Moderate Risk   (Orange)
  70-100 = Stable         (Green)
```

---

## Model Performance

### Text Module (Calibrated Voting Ensemble)
- **Components**: Logistic Regression, Linear SVM, Random Forest, Tuned LR
- **Accuracy**: 100% (on synthetic data; expect 85-95% on real Reddit data)
- **F1-Score**: 1.00
- **ROC-AUC**: 1.00

### Questionnaire Module (Calibrated Voting Ensemble)
- **Components**: Logistic Regression, Random Forest, Extra Trees, Tuned Gradient Boosting
- **Accuracy**: 96.75%
- **F1-Score**: 0.967
- **Per-class**: Low Risk 99%, Moderate 95%, High Risk 95%

---

## Alert System

| Alert Type | Trigger | Severity |
|-----------|---------|----------|
| Critical Score | `score < 40` | Critical/High |
| Sharp Drop | `drop > 15 pts in 3 days` | High |
| Sustained Decline | Negative trend slope over 5+ assessments | High |
| Module Red Flag | Any module score < 30 | Medium |

---

## Google Colab Guide

```python
# Upload project to Colab or clone from GitHub
# !git clone <repo-url>
# %cd "depression detection using text and speech"

# Install dependencies
# !pip install -r requirements.txt

# Upload real datasets
# from google.colab import files
# uploaded = files.upload()  # upload to data/raw/

# Train (GPU not needed for these models)
# !python train_all.py

# Evaluate
# !python evaluate_all.py

# Download trained models
# !zip -r models.zip models/
# files.download('models.zip')
```

Run Streamlit locally after copying models back.

---

## Implementation Methodology

1. **Literature Review & Gap Analysis** — Identified limitations of binary classification; defined continuous scoring as improvement target.
2. **Modular Architecture Design** — Separated text, questionnaire, and audio into independent pipelines with clean interfaces.
3. **Data Engineering** — Built configurable loaders with preprocessing, caching, and augmentation. Generated 1500+ synthetic samples.
4. **Feature Engineering** — TF-IDF with bigrams and sublinear TF for text; StandardScaler for tabular; MFCC for audio.
5. **Advanced Model Training** — Trained 4+ classifiers per module, compared via cross-validation, selected best ensemble with hyperparameter tuning.
6. **Probability Calibration** — Applied isotonic regression to ensure model confidence maps to true probability.
7. **Confidence-Weighted Fusion** — Implemented dynamic weighting where more certain modules get more influence.
8. **Trend & Alert System** — Built SQLite-backed time-series with moving averages, slope detection, and 4-level alerts.

---

## Limitations

- **Synthetic training data**: The bundled datasets are synthetic. Real-world performance depends on training with actual clinical or social media data.
- **Not clinically validated**: This is an educational decision-support tool, not a diagnostic instrument.
- **Text model overfitting**: Perfect accuracy on synthetic data indicates the model will need re-training on diverse real-world text.
- **Simple fusion**: Weighted averaging, even with confidence weighting, is less powerful than a learned meta-model.
- **No real-time audio**: Requires pre-recorded `.wav` files; live microphone not supported.
- **Single-user demo**: Uses a default user ID; multi-user auth would require additional infrastructure.

---

## Future Enhancements

- **Transformer-based NLP**: Replace TF-IDF with DistilBERT or MentalBERT for contextual text understanding
- **Learned fusion**: Train a neural meta-learner to optimally combine module outputs
- **Real-time voice analysis**: Add in-browser microphone recording with Web Audio API
- **Explainability (XAI)**: Integrate SHAP/LIME for per-prediction explanations
- **PHQ-9 / GAD-7 integration**: Support standardized clinical questionnaires
- **Mobile deployment**: Package models with ONNX for mobile inference
- **Federated learning**: Train across institutions without sharing sensitive data
- **Multi-user with auth**: Add login, per-user dashboards, and data isolation

---

## Viva Q&A Guide

### Why continuous scores instead of classification?

Classification groups all users into 2-3 buckets, losing critical nuance. A score of 42 and 68 would both be "moderate risk" in a 3-class system, but they represent very different mental states. Continuous scoring enables:
- **Trend detection**: You can see a user slowly declining even within the same risk band
- **Early warning**: A 10-point drop is actionable information that classification completely misses
- **Personalized thresholds**: Different users may have different baseline scores

### Why ensemble models instead of a single classifier?

No single algorithm is universally best. Our ensemble combines the strengths of different model families:
- **Logistic Regression**: Strong probabilistic baseline, handles high-dimensional sparse text well
- **SVM**: Excellent for binary classification with clear margins
- **Random Forest**: Captures non-linear interactions, robust to outliers
- **Gradient Boosting**: Iteratively corrects errors, often highest accuracy

The voting ensemble averages their predictions, reducing variance and improving generalization.

### What is probability calibration and why does it matter?

A model might predict P(distress) = 0.8, but that doesn't mean 80% of similar inputs are truly distress cases. Calibration (using isotonic regression) adjusts the predicted probabilities so they match true frequencies. This is critical because our fusion module uses these probabilities to compute scores — miscalibrated probabilities would produce misleading scores.

### How does confidence-weighted fusion work?

Each module produces both a score and a confidence value (how certain the model is). The fusion module multiplies the base weight by confidence:

```
effective_weight = base_weight * confidence
```

Then re-normalizes. This means if the text model is very confident (0.95) but the questionnaire model is uncertain (0.55), the text score gets proportionally more influence in the final result.

### What if audio is unavailable?

The system degrades gracefully. The text and questionnaire weights (0.4 each) are re-normalized to 0.5 each. The system works perfectly with just two modalities, and even with one if needed.

### How is this better than my previous project?

| Previous Project | This Project |
|-----------------|-------------|
| 4-second audio + few sentences | 1500+ text samples + 2000 survey responses |
| Single classifier | Calibrated ensemble of 4 classifiers |
| 3-class classification | Continuous 0-100 score |
| No history tracking | SQLite time-series with moving averages |
| No warnings | 4-level alert system with trend detection |
| Single modality | Multimodal with confidence-weighted fusion |
| Basic UI | Professional dashboard with gauge, radar, analytics |
| No evaluation artifacts | Confusion matrices, feature importance, scenario tests |
| No downloadable reports | Full assessment report export |

---

*Disclaimer: This system is an academic project and should not be used as a substitute for professional mental health care. If you or someone you know is in crisis, please contact emergency services or a mental health helpline.*
