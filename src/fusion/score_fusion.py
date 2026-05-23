"""
Advanced Multimodal Score Fusion Module.

Features:
  - Configurable weighted fusion with dynamic re-normalization
  - Confidence-weighted fusion: modules with higher model confidence
    get proportionally more weight
  - Risk-factor analysis explaining which modules drive the score
  - Score breakdown for explainability
"""

import os
import sys
from typing import Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def get_risk_band(score: float) -> str:
    """Map a 0-100 score to a human-readable risk band."""
    for band_name, (low, high) in config.RISK_BANDS.items():
        if low <= score <= high:
            return band_name
    return "unknown"


def get_risk_color(band: str) -> str:
    """Return a display color for the risk band."""
    return {"high_risk": "red", "moderate_risk": "orange", "stable": "green"}.get(band, "gray")


def get_risk_label(band: str) -> str:
    """Return a human-readable label."""
    return {"high_risk": "High Risk", "moderate_risk": "Moderate Risk", "stable": "Stable"}.get(band, "Unknown")


def fuse_scores(
    text_score: Optional[float] = None,
    questionnaire_score: Optional[float] = None,
    audio_score: Optional[float] = None,
    text_confidence: Optional[float] = None,
    questionnaire_confidence: Optional[float] = None,
    audio_confidence: Optional[float] = None,
    weights: Optional[Dict] = None,
    use_confidence_weighting: bool = True,
) -> dict:
    """
    Fuse individual module scores into a final mental health score.

    When confidence values are provided and use_confidence_weighting is True,
    the base weights are scaled by confidence:
        effective_weight[i] = base_weight[i] * confidence[i]
    Then re-normalized to sum to 1.

    Returns a rich result dict with score, risk info, breakdown, and explanations.
    """
    weights = weights or config.FUSION_CONFIG["weights"].copy()

    available = {}
    confidences = {}
    if text_score is not None:
        available["text"] = text_score
        confidences["text"] = text_confidence or 0.5
    if questionnaire_score is not None:
        available["questionnaire"] = questionnaire_score
        confidences["questionnaire"] = questionnaire_confidence or 0.5
    if audio_score is not None:
        available["audio"] = audio_score
        confidences["audio"] = audio_confidence or 0.5

    if not available:
        band = "moderate_risk"
        return {
            "final_score": 50.0,
            "risk_band": band,
            "risk_label": get_risk_label(band),
            "risk_color": get_risk_color(band),
            "module_scores": {},
            "weights_used": {},
            "risk_factors": [],
            "protective_factors": [],
            "explanation": "No module inputs provided.",
        }

    # Compute effective weights
    raw_weights = {k: weights.get(k, 0.0) for k in available}

    if use_confidence_weighting and any(c > 0 for c in confidences.values()):
        effective = {k: raw_weights[k] * confidences[k] for k in available}
    else:
        effective = raw_weights.copy()

    total = sum(effective.values())
    if total == 0:
        total = len(available)
        effective = {k: 1.0 for k in available}
    normalized = {k: v / total for k, v in effective.items()}

    # Compute final score
    final_score = sum(available[k] * normalized[k] for k in available)
    final_score = round(max(0.0, min(100.0, final_score)), 2)

    risk_band = get_risk_band(final_score)

    # Identify risk factors and protective factors
    risk_factors = []
    protective_factors = []
    for mod, score in available.items():
        mod_label = mod.replace("_", " ").title()
        if score < 40:
            risk_factors.append({"module": mod_label, "score": score, "severity": "high"})
        elif score < 60:
            risk_factors.append({"module": mod_label, "score": score, "severity": "moderate"})
        elif score >= 70:
            protective_factors.append({"module": mod_label, "score": score})

    # Build explanation
    explanations = []
    if risk_factors:
        high_risk_mods = [f["module"] for f in risk_factors if f["severity"] == "high"]
        mod_risk_mods = [f["module"] for f in risk_factors if f["severity"] == "moderate"]
        if high_risk_mods:
            explanations.append(f"High concern in: {', '.join(high_risk_mods)}")
        if mod_risk_mods:
            explanations.append(f"Moderate concern in: {', '.join(mod_risk_mods)}")
    if protective_factors:
        prot_mods = [f["module"] for f in protective_factors]
        explanations.append(f"Positive signals from: {', '.join(prot_mods)}")
    if not explanations:
        explanations.append("All modules show mixed signals.")

    return {
        "final_score": final_score,
        "risk_band": risk_band,
        "risk_label": get_risk_label(risk_band),
        "risk_color": get_risk_color(risk_band),
        "module_scores": {k: round(v, 2) for k, v in available.items()},
        "weights_used": {k: round(v, 4) for k, v in normalized.items()},
        "confidence_values": {k: round(v, 4) for k, v in confidences.items()},
        "risk_factors": risk_factors,
        "protective_factors": protective_factors,
        "explanation": " | ".join(explanations),
    }


if __name__ == "__main__":
    # Demo scenarios
    scenarios = [
        dict(text_score=85, questionnaire_score=90, text_confidence=0.92, questionnaire_confidence=0.88),
        dict(text_score=25, questionnaire_score=30, text_confidence=0.95, questionnaire_confidence=0.90),
        dict(text_score=80, questionnaire_score=30, text_confidence=0.60, questionnaire_confidence=0.95),
        dict(text_score=72, questionnaire_score=45, audio_score=60,
             text_confidence=0.85, questionnaire_confidence=0.80, audio_confidence=0.70),
    ]
    for i, s in enumerate(scenarios):
        r = fuse_scores(**s)
        print(f"Scenario {i+1}: Score={r['final_score']:.1f} ({r['risk_label']})")
        print(f"  Explanation: {r['explanation']}\n")
