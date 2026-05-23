"""
MindScore AI - Multimodal Mental Health Scoring System
======================================================
Futuristic dark-themed production dashboard.

Features:
  - Glassmorphism cards with neon accent glow
  - Animated score gauge with gradient arcs
  - Radar chart for multimodal breakdown
  - Audio waveform visualization with min/max duration check
  - Trend chart with moving averages and risk zones
  - Downloadable assessment reports
  - 4-page responsive layout
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import config
from src.fusion.score_fusion import fuse_scores, get_risk_label
from src.utils.trend_tracker import (
    save_score, get_history, compute_moving_average, detect_trend, get_statistics,
)
from src.utils.alert_system import check_alerts
from src.utils.scoring import score_to_description, score_to_emoji, generate_report

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScore AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Futuristic Dark Theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0a0a1a;
        --bg-card: rgba(15, 15, 35, 0.7);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.08);
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-cyan: #06b6d4;
        --accent-purple: #a855f7;
        --accent-green: #10b981;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        --glow-cyan: 0 0 20px rgba(6, 182, 212, 0.3);
        --glow-purple: 0 0 20px rgba(168, 85, 247, 0.3);
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 30%, #0a0a1a 60%, #121228 100%);
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d24 0%, #151538 50%, #0d0d24 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }

    /* Glass card */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(6, 182, 212, 0.2);
        box-shadow: var(--glow-cyan);
    }

    /* Hero title */
    .hero-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #06b6d4 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }

    .hero-sub {
        text-align: center;
        color: var(--text-muted);
        font-size: 1rem;
        font-weight: 400;
        margin-bottom: 2rem;
        letter-spacing: 0.05em;
    }

    /* Score display */
    .score-display {
        text-align: center;
        padding: 2rem 1rem;
    }
    .score-number {
        font-size: 5rem;
        font-weight: 900;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1;
        letter-spacing: -0.03em;
    }
    .score-suffix {
        font-size: 1.5rem;
        font-weight: 400;
        opacity: 0.5;
    }
    .score-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 0.8rem;
    }

    /* Module cards */
    .mod-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 14px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .mod-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--glow-purple);
    }
    .mod-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .mod-score {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
    }
    .mod-conf {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* Alert cards */
    .alert-card {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .alert-critical {
        background: rgba(239, 68, 68, 0.08);
        border-color: #ef4444;
    }
    .alert-high {
        background: rgba(245, 158, 11, 0.08);
        border-color: #f59e0b;
    }
    .alert-medium {
        background: rgba(6, 182, 212, 0.06);
        border-color: #06b6d4;
    }

    /* Tags */
    .tag-risk {
        display: inline-block;
        background: rgba(239, 68, 68, 0.12);
        color: #fca5a5;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.15rem;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .tag-safe {
        display: inline-block;
        background: rgba(16, 185, 129, 0.12);
        color: #6ee7b7;
        padding: 0.3rem 0.8rem;
        border-radius: 50px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.15rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    /* Stat card */
    .stat-card {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-cyan);
    }
    .stat-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* Audio viz area */
    .audio-info {
        background: rgba(168, 85, 247, 0.06);
        border: 1px solid rgba(168, 85, 247, 0.15);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }

    /* Disclaimer */
    .disclaimer {
        background: var(--bg-glass);
        border: 1px solid var(--border-glass);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        font-size: 0.72rem;
        color: var(--text-muted);
        text-align: center;
        margin-top: 1.5rem;
    }

    /* Override Streamlit elements for dark theme */
    .stTextArea textarea, .stTextInput input {
        background: rgba(15, 15, 35, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 1px var(--accent-cyan) !important;
    }
    .stSlider > div > div {
        color: var(--text-primary) !important;
    }
    div[data-testid="stExpander"] {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border-glass) !important;
        border-radius: 12px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-glass);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid var(--border-glass);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: var(--text-secondary) !important;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(6, 182, 212, 0.15) !important;
        color: var(--accent-cyan) !important;
    }
    .stDownloadButton button {
        background: linear-gradient(135deg, #06b6d4, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #06b6d4 0%, #a855f7 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.75rem 2rem !important;
        letter-spacing: 0.03em;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 30px rgba(6, 182, 212, 0.4) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _models_ready():
    return {
        "text": os.path.exists(config.TEXT_CONFIG["model_path"]),
        "questionnaire": os.path.exists(config.QUESTIONNAIRE_CONFIG["model_path"]),
        "audio": os.path.exists(config.AUDIO_CONFIG["model_path"]) and config.AUDIO_CONFIG["enabled"],
    }


def _score_color(score):
    if score >= 70: return "#10b981"
    elif score >= 40: return "#f59e0b"
    else: return "#ef4444"


def _score_glow(score):
    c = _score_color(score)
    return f"0 0 40px {c}66, 0 0 80px {c}33"


def _gauge(score, risk_color):
    color = {"green": "#10b981", "orange": "#f59e0b", "red": "#ef4444"}.get(risk_color, "#06b6d4")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "", "font": {"size": 52, "color": color, "family": "JetBrains Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155",
                     "dtick": 20, "tickfont": {"size": 11, "color": "#64748b"}},
            "bar": {"color": color, "thickness": 0.8},
            "bgcolor": "rgba(255,255,255,0.02)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 39], "color": "rgba(239,68,68,0.06)"},
                {"range": [40, 69], "color": "rgba(245,158,11,0.04)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.04)"},
            ],
            "threshold": {"line": {"color": "#ef4444", "width": 2}, "thickness": 0.8, "value": 40},
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=25, r=25, t=35, b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    return fig


def _radar(module_scores):
    labels = [k.replace("_", " ").title() for k in module_scores.keys()]
    values = list(module_scores.values())
    labels.append(labels[0])
    values.append(values[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        fillcolor="rgba(6, 182, 212, 0.12)",
        line=dict(color="#06b6d4", width=2.5),
        marker=dict(size=7, color="#06b6d4"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.05)",
                            tickfont=dict(size=9, color="#64748b")),
            angularaxis=dict(tickfont=dict(size=12, color="#94a3b8"),
                             gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, height=300,
        margin=dict(l=55, r=55, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _trend_chart(history):
    history = history.sort_values("timestamp")
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=39, fillcolor="rgba(239,68,68,0.04)", line_width=0)
    fig.add_hrect(y0=40, y1=69, fillcolor="rgba(245,158,11,0.02)", line_width=0)
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(16,185,129,0.02)", line_width=0)
    fig.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.3)", line_width=1)

    colors = [_score_color(s) for s in history["final_score"]]
    fig.add_trace(go.Scatter(
        x=history["timestamp"], y=history["final_score"],
        mode="lines+markers", name="Score",
        line=dict(color="#06b6d4", width=2.5),
        marker=dict(size=9, color=colors, line=dict(width=2, color="#0a0a1a")),
    ))
    if len(history) >= 3:
        ma = compute_moving_average(history, window=3)
        fig.add_trace(go.Scatter(
            x=history.sort_values("timestamp")["timestamp"], y=ma.values,
            mode="lines", name="Moving Avg",
            line=dict(color="#a855f7", width=2, dash="dot"), opacity=0.7,
        ))
    fig.update_layout(
        xaxis_title="", yaxis_title="Score", yaxis=dict(range=[0, 105]),
        template="plotly_dark", height=420,
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center", font=dict(size=11)),
        margin=dict(l=40, r=15, t=15, b=35),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
        yaxis2=dict(gridcolor="rgba(255,255,255,0.03)"),
    )
    return fig


def _audio_waveform(audio_path, sr=22050):
    """Create a waveform visualization of the uploaded audio."""
    try:
        import librosa
        y, _ = librosa.load(audio_path, sr=sr)
        duration = len(y) / sr
        # Downsample for plotting
        step = max(1, len(y) // 2000)
        y_plot = y[::step]
        t = np.linspace(0, duration, len(y_plot))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=y_plot, mode="lines",
            line=dict(color="#a855f7", width=1),
            fill="tozeroy", fillcolor="rgba(168, 85, 247, 0.1)",
        ))
        fig.update_layout(
            height=150, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, title="Time (s)", titlefont=dict(size=10, color="#64748b"),
                       tickfont=dict(size=9, color="#64748b")),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False,
        )
        return fig, duration
    except Exception:
        return None, 0


def _audio_duration_label(duration_sec: float) -> tuple[str, str]:
    """Return (status, html_message) for min/max duration from config."""
    min_dur = config.AUDIO_CONFIG.get("min_duration_sec", 10)
    max_dur = config.AUDIO_CONFIG.get("max_duration_sec")
    range_txt = f"{min_dur}–{max_dur}s" if max_dur else f"at least {min_dur}s"

    if duration_sec < min_dur:
        return "short", (
            f'<span style="color:#ef4444;font-size:0.85rem;">'
            f"Duration: {duration_sec:.1f}s | Too short (need {range_txt})</span>"
        )
    if max_dur and duration_sec > max_dur:
        return "long", (
            f'<span style="color:#ef4444;font-size:0.85rem;">'
            f"Duration: {duration_sec:.1f}s | Too long (maximum {max_dur}s)</span>"
        )
    return "ok", (
        f'<span style="color:#10b981;font-size:0.85rem;">'
        f"Duration: {duration_sec:.1f}s | Ready for analysis ({range_txt})</span>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧠 MindScore AI")
    st.caption("v2.0 | Multimodal Assessment Engine")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Assessment", "📈 Trends", "📊 Analytics", "ℹ️ System"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**System Status**")
    status = _models_ready()
    for mod, ready in status.items():
        icon = "🟢" if ready else "🔴"
        st.markdown(f"<span style='font-size:0.85rem;'>{icon} {mod.title()} Engine</span>",
                    unsafe_allow_html=True)

    if not all(status.values()):
        st.warning("Run `python train_all.py` to train all models.")

    st.divider()
    st.caption("Built with scikit-learn + Streamlit")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
if "🏠" in page:
    st.markdown('<h1 class="hero-title">Mental Health Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">AI-powered multimodal analysis | Continuous 0-100 wellness score</p>',
                unsafe_allow_html=True)

    tab_text, tab_quest, tab_audio = st.tabs(["📝 Text Analysis", "📋 Questionnaire", "🎙️ Voice Analysis"])

    # ── TEXT TAB ─────────────────────────────────────────────────────────────
    with tab_text:
        st.markdown("##### Share how you've been feeling")
        st.caption("Write at least 2-3 detailed sentences for accurate analysis. "
                   "Describe your mood, sleep, energy, social life, or any concerns.")
        user_text = st.text_area(
            "entry", height=220, label_visibility="collapsed",
            placeholder=(
                "I've been feeling really overwhelmed lately with everything going on. "
                "I can't seem to concentrate and my sleep has been terrible for weeks. "
                "I keep cancelling plans with friends because I just don't have the energy..."
            ),
        )
        if user_text:
            wc = len(user_text.split())
            q = "Excellent" if wc >= 30 else ("Good" if wc >= 15 else ("Fair" if wc >= 5 else "Too short"))
            c = "#10b981" if wc >= 15 else ("#f59e0b" if wc >= 5 else "#ef4444")
            st.markdown(f"<span style='color:{c};font-size:0.8rem;font-family:JetBrains Mono;'>"
                        f"{wc} words | {q}</span>", unsafe_allow_html=True)

    # ── QUESTIONNAIRE TAB ────────────────────────────────────────────────────
    with tab_quest:
        st.markdown("##### Rate each aspect of your recent wellbeing")
        st.caption("1 = Very poor/low  |  10 = Excellent/high (stress: 1=low, 10=high)")
        q_responses = {}
        cols = st.columns(2)
        for i, item in enumerate(config.QUESTIONNAIRE_ITEMS):
            with cols[i % 2]:
                q_responses[item["id"]] = st.slider(
                    item["question"], min_value=item["min"], max_value=item["max"],
                    value=5, key=f"q_{item['id']}",
                )

    # ── AUDIO TAB ────────────────────────────────────────────────────────────
    with tab_audio:
        _min_d = config.AUDIO_CONFIG.get("min_duration_sec", 10)
        _max_d = config.AUDIO_CONFIG.get("max_duration_sec")
        _range_hint = f"{_min_d}–{_max_d} seconds" if _max_d else f"at least {_min_d} seconds"
        st.markdown("##### Record or upload voice")
        st.markdown(
            '<div class="audio-info">'
            f'<strong>Requirements:</strong> {_range_hint} | speak clearly about how you feel<br>'
            '<span style="color:#94a3b8;font-size:0.8rem;">'
            'Use live recording (recommended) or upload a .wav file.'
            '</span></div>',
            unsafe_allow_html=True,
        )
        st.write("")

        col_rec, col_up = st.columns(2)
        with col_rec:
            st.markdown("**Live recording**")
            mic_audio = st.audio_input(
                "Click to record (allow microphone access)",
                key="mic_input",
            )
        with col_up:
            st.markdown("**Or upload file**")
            upload_audio = st.file_uploader(
                "Upload .wav",
                type=["wav"],
                key="wav_upload",
                label_visibility="collapsed",
            )

        # One source for the rest of the app: mic wins if both exist
        audio_file = mic_audio if mic_audio is not None else upload_audio

        if audio_file is not None:
            st.audio(audio_file)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_file.seek(0)
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            audio_file.seek(0)

            waveform_fig, audio_dur = _audio_waveform(tmp_path)
            if waveform_fig:
                st.plotly_chart(waveform_fig, use_container_width=True)
                _, dur_msg = _audio_duration_label(audio_dur)
                st.markdown(dur_msg, unsafe_allow_html=True)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        else:
            st.info(
                f"Record {_range_hint} of audio, or upload a .wav file, then click **Run Assessment**."
            )

    # ── COMPUTE ──────────────────────────────────────────────────────────────
    st.write("")
    st.write("")
    compute = st.button("Run Assessment", type="primary", use_container_width=True)

    if compute:
        progress = st.progress(0, text="Initializing analysis...")
        text_score = text_conf = q_score = q_conf = audio_score = audio_conf = None
        text_details = q_details = audio_details = {}

        # --- Text ---
        progress.progress(10, text="Analyzing text patterns...")
        if user_text.strip() and status["text"]:
            try:
                from src.inference.text_inference import predict_score as text_predict
                text_result = text_predict(user_text)
                text_score = text_result["score"]
                text_conf = text_result.get("confidence")
                text_details = text_result
            except Exception as e:
                st.error(f"Text module: {e}")

        # --- Questionnaire ---
        progress.progress(40, text="Processing questionnaire responses...")
        if status["questionnaire"]:
            try:
                from src.inference.questionnaire_inference import predict_score as q_predict
                q_result = q_predict(q_responses)
                q_score = q_result["score"]
                q_conf = q_result.get("confidence")
                q_details = q_result
            except Exception as e:
                st.error(f"Questionnaire module: {e}")

        # --- Audio ---
        progress.progress(60, text="Analyzing vocal patterns...")
        if audio_file is not None and status["audio"]:
            try:
                from src.inference.audio_inference import predict_score as audio_predict
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    audio_file.seek(0)
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                audio_result = audio_predict(tmp_path)
                if audio_result.get("available"):
                    audio_score = audio_result["score"]
                    audio_conf = audio_result.get("confidence")
                    audio_details = audio_result
                elif audio_result.get("error"):
                    st.warning(audio_result["error"])
                os.unlink(tmp_path)
            except Exception as e:
                st.error(f"Audio module: {e}")

        # --- Fusion ---
        progress.progress(80, text="Fusing multimodal signals...")
        fusion = fuse_scores(
            text_score=text_score, questionnaire_score=q_score, audio_score=audio_score,
            text_confidence=text_conf, questionnaire_confidence=q_conf, audio_confidence=audio_conf,
        )
        final_score = fusion["final_score"]
        risk_band = fusion["risk_band"]
        risk_color = fusion["risk_color"]
        risk_label = fusion["risk_label"]

        # --- Alerts ---
        mod_map = {"text": text_score, "questionnaire": q_score, "audio": audio_score}
        alerts = check_alerts(current_score=final_score, module_scores=mod_map)

        # --- Save ---
        save_score(
            final_score=final_score, text_score=text_score,
            questionnaire_score=q_score, audio_score=audio_score,
            risk_band=risk_band, text_confidence=text_conf,
            quest_confidence=q_conf, audio_confidence=audio_conf,
        )
        progress.progress(100, text="Assessment complete")

        # ── RESULTS ──────────────────────────────────────────────────────────
        st.write("")
        st.markdown("---")

        # Score + Gauge + Radar
        col_score, col_gauge, col_radar = st.columns([2, 3, 3])

        with col_score:
            sc = _score_color(final_score)
            badge_bg = f"{sc}22"
            st.markdown(
                f'<div class="score-display">'
                f'<div class="score-number" style="color:{sc};text-shadow:{_score_glow(final_score)};">'
                f'{final_score:.0f}<span class="score-suffix">/100</span></div>'
                f'<div class="score-badge" style="background:{badge_bg};color:{sc};border:1px solid {sc}44;">'
                f'{score_to_emoji(final_score)} {risk_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_gauge:
            st.plotly_chart(_gauge(final_score, risk_color), use_container_width=True)

        with col_radar:
            active = {k: v for k, v in fusion["module_scores"].items() if v is not None}
            if len(active) >= 2:
                st.plotly_chart(_radar(active), use_container_width=True)

        # Interpretation
        st.markdown(f"<div class='glass-card' style='text-align:center;'>"
                    f"<span style='color:{_score_color(final_score)};'>"
                    f"{score_to_description(final_score)}</span></div>",
                    unsafe_allow_html=True)

        # Module cards
        st.write("")
        mod_data = [
            ("Text Analysis", "📝", text_score, text_conf, text_details),
            ("Questionnaire", "📋", q_score, q_conf, q_details),
            ("Voice Analysis", "🎙️", audio_score, audio_conf, audio_details),
        ]
        mc = st.columns(3)
        for col, (name, icon, sc, conf, det) in zip(mc, mod_data):
            with col:
                if sc is not None:
                    c = _score_color(sc)
                    conf_str = f"{conf:.0%}" if conf else "N/A"
                    st.markdown(
                        f'<div class="mod-card">'
                        f'<div class="mod-label">{icon} {name}</div>'
                        f'<div class="mod-score" style="color:{c};">{sc:.1f}</div>'
                        f'<div class="mod-conf">Confidence: {conf_str}</div>'
                        f'</div>', unsafe_allow_html=True,
                    )
                    with st.expander("Details"):
                        st.json(det)
                else:
                    st.markdown(
                        f'<div class="mod-card">'
                        f'<div class="mod-label">{icon} {name}</div>'
                        f'<div class="mod-score" style="color:#334155;">--</div>'
                        f'<div class="mod-conf">Not active</div>'
                        f'</div>', unsafe_allow_html=True,
                    )

        # Risk / Protective factors
        if fusion.get("risk_factors") or fusion.get("protective_factors"):
            st.write("")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**Risk Factors**")
                if fusion["risk_factors"]:
                    tags = ""
                    for rf in fusion["risk_factors"]:
                        sev = "🔴" if rf["severity"] == "high" else "🟡"
                        tags += f'<span class="tag-risk">{sev} {rf["module"]}: {rf["score"]:.0f}</span> '
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown('<span class="tag-safe">No significant risks detected</span>',
                                unsafe_allow_html=True)
            with fc2:
                st.markdown("**Protective Factors**")
                if fusion["protective_factors"]:
                    tags = ""
                    for pf in fusion["protective_factors"]:
                        tags += f'<span class="tag-safe">🟢 {pf["module"]}: {pf["score"]:.0f}</span> '
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.caption("None identified in this assessment.")

        # Alerts
        if alerts:
            st.write("")
            st.markdown("**Alerts & Recommendations**")
            for a in alerts:
                css = f"alert-{a['severity']}" if a["severity"] in ("critical","high","medium") else "alert-medium"
                icons = {"critical":"🚨","high":"⚠️","medium":"💡","info":"ℹ️"}
                st.markdown(
                    f'<div class="alert-card {css}">'
                    f'<strong>{icons.get(a["severity"],"📌")} {a["title"]}</strong><br>'
                    f'<span style="color:#cbd5e1;">{a["message"]}</span><br>'
                    f'<em style="color:#94a3b8;font-size:0.85rem;">{a.get("advice","")}</em>'
                    f'</div>', unsafe_allow_html=True,
                )

        # Technical details
        with st.expander("Technical Details"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Fusion Weights**")
                st.json(fusion["weights_used"])
            with c2:
                st.markdown("**Confidence Values**")
                st.json(fusion.get("confidence_values", {}))

        # Download report
        st.write("")
        report_text = generate_report(fusion, alerts)
        st.download_button(
            "Download Assessment Report",
            data=report_text,
            file_name=f"mindscore_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain", use_container_width=True,
        )

        st.markdown(
            '<div class="disclaimer">'
            'This is an AI screening tool for educational purposes. '
            'It does NOT replace professional mental health evaluation. '
            'If you are in crisis, please contact emergency services or a helpline.'
            '</div>', unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif "📈" in page:
    st.markdown('<h1 class="hero-title">Score Trends</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Track your mental health trajectory over time</p>',
                unsafe_allow_html=True)

    history = get_history(limit=500)
    if history.empty:
        st.info("Complete your first assessment to see trends here.")
    else:
        history = history.sort_values("timestamp")
        fc, _, tc = st.columns([2, 1, 2])
        with fc:
            rng = st.selectbox("Time Range", ["Last 7 days", "Last 30 days", "All time"])
        if rng == "Last 7 days":
            history = history[history["timestamp"] >= datetime.now() - timedelta(days=7)]
        elif rng == "Last 30 days":
            history = history[history["timestamp"] >= datetime.now() - timedelta(days=30)]

        if history.empty:
            st.info("No data in selected range.")
        else:
            trend = detect_trend(history, window=5)
            with tc:
                t_icons = {"improving": "📈", "declining": "📉", "stable": "➡️"}
                t_colors = {"improving": "#10b981", "declining": "#ef4444", "stable": "#06b6d4"}
                st.markdown(
                    f'<div style="text-align:right;padding-top:1.2rem;">'
                    f'<span style="font-size:1.6rem;">{t_icons.get(trend["direction"],"")}</span> '
                    f'<span style="color:{t_colors.get(trend["direction"],"#94a3b8")};font-weight:700;'
                    f'font-size:1.1rem;">{trend["direction"].title()}</span></div>',
                    unsafe_allow_html=True,
                )

            st.plotly_chart(_trend_chart(history), use_container_width=True)
            st.info(trend["interpretation"])

            stats = get_statistics()
            if stats:
                sc = st.columns(5)
                items = [
                    ("Latest", stats.get("latest_score", 0)),
                    ("Average", stats.get("mean_score", 0)),
                    ("Best", stats.get("max_score", 0)),
                    ("Lowest", stats.get("min_score", 0)),
                    ("Total", stats.get("total_assessments", 0)),
                ]
                for col, (lbl, val) in zip(sc, items):
                    with col:
                        v = f"{val:.1f}" if isinstance(val, float) else str(val)
                        st.markdown(
                            f'<div class="stat-card">'
                            f'<div class="stat-value">{v}</div>'
                            f'<div class="stat-label">{lbl}</div>'
                            f'</div>', unsafe_allow_html=True,
                        )

            with st.expander("Raw Data"):
                disp = history[["timestamp","final_score","text_score",
                                "questionnaire_score","audio_score","risk_band"]].copy()
                disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
                st.dataframe(disp, use_container_width=True)

            csv = history.to_csv(index=False)
            st.download_button("Export CSV", csv, "score_history.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "📊" in page:
    st.markdown('<h1 class="hero-title">Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Deep insights into your assessment patterns</p>',
                unsafe_allow_html=True)

    history = get_history(limit=500)
    if history.empty or len(history) < 2:
        st.info("Complete at least 2 assessments to unlock analytics.")
    else:
        history = history.sort_values("timestamp")

        st.markdown("### Module Comparison")
        mf = go.Figure()
        for col, color, name in [
            ("text_score", "#06b6d4", "Text"),
            ("questionnaire_score", "#a855f7", "Questionnaire"),
            ("audio_score", "#10b981", "Audio"),
            ("final_score", "#f59e0b", "Final"),
        ]:
            valid = history[col].dropna()
            if not valid.empty:
                mf.add_trace(go.Scatter(
                    x=history.loc[valid.index, "timestamp"], y=valid,
                    mode="lines+markers", name=name,
                    line=dict(width=2.5, color=color), marker=dict(size=5),
                ))
        mf.update_layout(
            height=380, template="plotly_dark",
            yaxis=dict(range=[0, 105], title="Score"),
            margin=dict(l=40, r=15, t=15, b=35),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(mf, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Risk Distribution")
            bands = history["risk_band"].value_counts()
            bc = {"high_risk":"#ef4444","moderate_risk":"#f59e0b","stable":"#10b981"}
            fig_pie = go.Figure(go.Pie(
                labels=[b.replace("_"," ").title() for b in bands.index],
                values=bands.values,
                marker=dict(colors=[bc.get(b,"#64748b") for b in bands.index]),
                hole=0.5, textinfo="label+percent",
                textfont=dict(color="#e2e8f0"),
            ))
            fig_pie.update_layout(
                height=320, margin=dict(l=10,r=10,t=10,b=10),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown("### Score Distribution")
            fig_hist = px.histogram(
                history, x="final_score", nbins=15,
                color_discrete_sequence=["#06b6d4"],
            )
            fig_hist.update_layout(
                height=320, template="plotly_dark",
                margin=dict(l=20,r=10,t=10,b=30),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Score", yaxis_title="Count",
            )
            st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SYSTEM / ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif "ℹ️" in page:
    st.markdown('<h1 class="hero-title">System Architecture</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Technical specifications and methodology</p>',
                unsafe_allow_html=True)

    st.markdown("""
<div class="glass-card">

### Architecture Overview

| Module | Input | Model | Features | Output |
|--------|-------|-------|----------|--------|
| **Text Engine** | Free-form text | Calibrated Voting Ensemble (LR + SVM + RF + GB) | TF-IDF 5000 features, bigrams | Score + Confidence |
| **Questionnaire Engine** | 10-item survey | Calibrated Voting Ensemble (LR + RF + ET + GB) | StandardScaler, 10 features | Score + Confidence |
| **Voice Engine** | 10s+ audio (.wav) | Calibrated Ensemble (SVM + RF + GB) | 34-dim: MFCC + spectral + ZCR | Score + Confidence |
| **Fusion Engine** | Module outputs | Confidence-weighted averaging | Dynamic re-normalization | Final 0-100 score |

</div>

<div class="glass-card">

### Scoring Pipeline

Each module produces a calibrated probability distribution.
Probabilities are converted to a 0-100 score using module-specific mappings.
The fusion engine applies confidence-weighted averaging with dynamic weight re-normalization.

**Alert System:** 4-level detection (critical score, sharp drop, sustained decline, module red flags)

**Trend Analysis:** Linear regression slope detection + 3-point moving average

</div>

<div class="glass-card">

### Risk Bands

| Score | Level | Indicator |
|-------|-------|-----------|
| 0-39 | High Risk | Significant distress indicators |
| 40-69 | Moderate Risk | Some concerning signals |
| 70-100 | Stable | Positive wellness indicators |

</div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="disclaimer">'
        'MindScore AI is an academic project. It does NOT replace professional mental health care.'
        '</div>', unsafe_allow_html=True,
    )
