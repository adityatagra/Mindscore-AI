"""
Advanced Alert System.

Multi-level alert detection:
  1. Critical Score: score below threshold
  2. Sharp Drop: large decrease within time window
  3. Sustained Decline: consistently declining trend over multiple assessments
  4. Module-specific alerts: individual modules showing red flags

Each alert has a severity (critical, high, medium, info) and actionable advice.
"""

import os
import sys
from typing import List, Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config
from src.utils.trend_tracker import get_recent_scores, detect_trend, get_history


def check_alerts(
    current_score: float,
    module_scores: dict = None,
    user_id: str = "default_user",
    alert_cfg: dict = None,
) -> List[Dict]:
    """
    Evaluate current state for all alert conditions.

    Returns list of alert dicts sorted by severity.
    """
    alert_cfg = alert_cfg or config.ALERT_CONFIG
    alerts = []
    module_scores = module_scores or {}

    # --- Alert 1: Critical Score ---
    if current_score < alert_cfg["critical_score_threshold"]:
        severity = "critical" if current_score < 25 else "high"
        alerts.append({
            "type": "critical_score",
            "severity": severity,
            "title": "Critical Mental Health Score",
            "message": (
                f"Your score ({current_score:.1f}/100) is below the safety threshold "
                f"of {alert_cfg['critical_score_threshold']}. This indicates significant distress."
            ),
            "advice": (
                "Please consider reaching out to a mental health professional, "
                "a trusted friend or family member, or a crisis helpline. "
                "You are not alone and help is available."
            ),
            "details": {"current_score": current_score, "threshold": alert_cfg["critical_score_threshold"]},
        })

    # --- Alert 2: Sharp Drop ---
    window_days = alert_cfg["drop_window_days"]
    history = get_recent_scores(user_id=user_id, days=window_days)

    if not history.empty and len(history) >= 2:
        max_recent = float(history["final_score"].max())
        drop = max_recent - current_score
        if drop >= alert_cfg["drop_threshold"]:
            severity = "critical" if drop >= alert_cfg["drop_threshold"] * 2 else "high"
            alerts.append({
                "type": "sharp_drop",
                "severity": severity,
                "title": "Rapid Score Decline Detected",
                "message": (
                    f"Your score has dropped by {drop:.1f} points in the last {window_days} days "
                    f"(from {max_recent:.1f} to {current_score:.1f})."
                ),
                "advice": (
                    "A significant decline may indicate worsening mental health. "
                    "Consider what may have changed recently and whether you need additional support."
                ),
                "details": {"current": current_score, "previous_max": max_recent, "drop": round(drop, 2)},
            })

    # --- Alert 3: Sustained Decline ---
    full_history = get_history(user_id=user_id, limit=10)
    if len(full_history) >= 4:
        trend = detect_trend(full_history, window=5)
        if trend["direction"] == "declining" and trend["slope"] < -3.0:
            alerts.append({
                "type": "sustained_decline",
                "severity": "high",
                "title": "Sustained Declining Trend",
                "message": (
                    f"Your scores have been consistently declining over recent assessments "
                    f"(trend slope: {trend['slope']:.1f} per assessment)."
                ),
                "advice": (
                    "A persistent downward trend suggests your mental health may be worsening. "
                    "Early intervention is most effective. Please consider professional support."
                ),
                "details": trend,
            })

    # --- Alert 4: Module-Specific Red Flags ---
    for mod_name, mod_score in module_scores.items():
        if mod_score is not None and mod_score < 30:
            alerts.append({
                "type": "module_alert",
                "severity": "medium",
                "title": f"Low {mod_name.replace('_', ' ').title()} Score",
                "message": (
                    f"Your {mod_name.replace('_', ' ')} analysis shows a concerning score "
                    f"of {mod_score:.1f}/100."
                ),
                "advice": f"The {mod_name.replace('_', ' ')} module detected significant indicators of distress.",
                "details": {"module": mod_name, "score": mod_score},
            })

    # Sort by severity
    severity_order = {"critical": 0, "high": 1, "medium": 2, "info": 3}
    alerts.sort(key=lambda a: severity_order.get(a["severity"], 99))

    return alerts


def format_alerts_for_display(alerts: List[Dict]) -> str:
    """Format alerts as readable text."""
    if not alerts:
        return ""
    lines = []
    icons = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "info": "[i]"}
    for a in alerts:
        icon = icons.get(a["severity"], "[?]")
        lines.append(f"{icon} {a['title']}")
        lines.append(f"    {a['message']}")
        if a.get("advice"):
            lines.append(f"    -> {a['advice']}")
        lines.append("")
    return "\n".join(lines)
