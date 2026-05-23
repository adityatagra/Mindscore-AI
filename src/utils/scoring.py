"""
Scoring Utilities.

Provides score interpretation, descriptions, report generation,
and helper functions used across the pipeline.
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def score_to_description(score: float) -> str:
    """Return a detailed description for a given score."""
    if score >= 85:
        return (
            "Your mental health indicators are excellent. You show strong positive signals "
            "across all assessed dimensions. Continue maintaining your healthy routines."
        )
    elif score >= 70:
        return (
            "Your mental health appears stable and generally positive. Minor stressors "
            "may be present but you seem to be coping well overall."
        )
    elif score >= 55:
        return (
            "Moderate risk indicators detected. You may be experiencing some stress or "
            "emotional difficulty. Consider practicing stress-management techniques, "
            "maintaining social connections, and prioritizing self-care."
        )
    elif score >= 40:
        return (
            "Elevated risk indicators present. Your responses suggest you may be going "
            "through a difficult period. It may help to talk to someone you trust, "
            "a counselor, or a mental health professional."
        )
    elif score >= 25:
        return (
            "High-risk indicators detected across multiple dimensions. We strongly "
            "recommend reaching out to a mental health professional. You don't have "
            "to face this alone."
        )
    else:
        return (
            "Critical risk level detected. Your responses indicate significant distress. "
            "Please reach out to a crisis helpline, a mental health professional, or "
            "someone you trust immediately. Help is available and you matter."
        )


def score_to_emoji(score: float) -> str:
    """Return a colored circle emoji for the score level."""
    if score >= 70:
        return "🟢"
    elif score >= 40:
        return "🟡"
    else:
        return "🔴"


def generate_report(fusion_result: dict, alerts: list = None) -> str:
    """Generate a comprehensive text report for export or display."""
    score = fusion_result["final_score"]
    band = fusion_result.get("risk_label", fusion_result["risk_band"].replace("_", " ").title())
    modules = fusion_result.get("module_scores", {})
    weights = fusion_result.get("weights_used", {})
    risk_factors = fusion_result.get("risk_factors", [])
    protective = fusion_result.get("protective_factors", [])

    lines = [
        "=" * 60,
        "  MENTAL HEALTH ASSESSMENT REPORT",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        f"  OVERALL SCORE:  {score:.1f} / 100",
        f"  RISK LEVEL:     {band}",
        f"  STATUS:         {score_to_emoji(score)}",
        "",
        "-" * 60,
        "  MODULE BREAKDOWN",
        "-" * 60,
    ]

    for mod, val in modules.items():
        weight = weights.get(mod, 0)
        lines.append(f"    {mod.replace('_', ' ').title():<25s}: {val:6.1f}  (weight: {weight:.1%})")

    lines.extend(["", "-" * 60, "  INTERPRETATION", "-" * 60, ""])
    lines.append(f"  {score_to_description(score)}")

    if risk_factors:
        lines.extend(["", "  RISK FACTORS:"])
        for rf in risk_factors:
            lines.append(f"    - {rf['module']}: Score {rf['score']:.1f} ({rf['severity']} concern)")

    if protective:
        lines.extend(["", "  PROTECTIVE FACTORS:"])
        for pf in protective:
            lines.append(f"    + {pf['module']}: Score {pf['score']:.1f}")

    if alerts:
        lines.extend(["", "-" * 60, "  ALERTS", "-" * 60, ""])
        for a in alerts:
            lines.append(f"  [{a['severity'].upper()}] {a['title']}")
            lines.append(f"    {a['message']}")
            if a.get("advice"):
                lines.append(f"    Recommendation: {a['advice']}")
            lines.append("")

    lines.extend([
        "",
        "=" * 60,
        "  DISCLAIMER",
        "  This is an AI-based screening tool and does NOT replace",
        "  professional mental health evaluation or diagnosis.",
        "=" * 60,
    ])

    return "\n".join(lines)
