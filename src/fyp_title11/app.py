from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from audiorecorder import audiorecorder

matplotlib.use("Agg")

# ─── Path Setup ───────────────────────────────────────────────────────────────
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
src_path = project_root / "src"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# ─── Imports ──────────────────────────────────────────────────────────────────
try:
    from scripts.hybrid_retrieval import (
        SR,
        discover_references,
        downsample_chroma,
        dtw_chroma_distance,
        minmax_normalize,
        safe_extract_query_features,
        safe_extract_reference_features,
        subseq_dtw_distance,
    )
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# ─── Retrieval Settings ───────────────────────────────────────────────────────
RETRIEVAL_MODE_LABEL = "Vocal-only DTW Retrieval"

# Demo setting: faster than 20, still reasonable
SHORTLIST_SIZE = 10
TOP_K_DISPLAY = 3

# Exact match only when clearly dominant
EXACT_SCORE_THRESHOLD = 0.18
EXACT_GAP_THRESHOLD = 0.10

VOCAL_ONLY_WEIGHTS = {
    "vocal_pitch": 0.60,
    "vocal_chroma": 0.40,
}

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hum2Tune",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0a0a0f;
    --surface:  #12121a;
    --card:     #1a1a26;
    --border:   #2a2a3d;
    --accent:   #6c63ff;
    --accent2:  #ff6584;
    --accent3:  #43e97b;
    --text:     #f0f0f8;
    --muted:    #7a7a9a;
    --success:  #43e97b;
    --warning:  #f7b731;
    --danger:   #fc5c65;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp {
    background: radial-gradient(ellipse at 20% 0%, #1a0a2e 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 100%, #0a1a2e 0%, transparent 60%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1rem 4rem 1rem; max-width: 760px; }

.hero { text-align:center; padding:3rem 1rem 2rem; }
.hero-icon {
    font-size:4rem; display:block; margin-bottom:0.5rem;
    filter:drop-shadow(0 0 30px #6c63ff88);
    animation:pulse-glow 3s ease-in-out infinite;
}
@keyframes pulse-glow {
    0%,100% { filter:drop-shadow(0 0 20px #6c63ff66); }
    50%      { filter:drop-shadow(0 0 50px #6c63ffcc); }
}
.hero-title {
    font-family:'Syne',sans-serif; font-size:3.2rem; font-weight:800;
    background:linear-gradient(135deg,#6c63ff,#ff6584,#43e97b);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1; margin:0; letter-spacing:-1px;
}
.hero-subtitle { font-size:1.05rem; color:var(--muted); margin-top:0.75rem; font-weight:300; }
.hero-badge {
    display:inline-block; background:rgba(108,99,255,0.15);
    border:1px solid rgba(108,99,255,0.4); border-radius:100px;
    padding:0.25rem 1rem; font-size:0.78rem; color:#a09aff;
    margin-top:1rem; letter-spacing:1px; text-transform:uppercase;
}

.card {
    background:var(--card); border:1px solid var(--border);
    border-radius:20px; padding:1.75rem; margin:1rem 0; position:relative; overflow:hidden;
}
.card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:1px;
    background:linear-gradient(90deg,transparent,rgba(108,99,255,0.5),transparent);
}
.section-label {
    font-family:'Syne',sans-serif; font-size:0.72rem; letter-spacing:3px;
    text-transform:uppercase; color:var(--accent); margin-bottom:1rem;
    display:flex; align-items:center; gap:0.5rem;
}
.section-label::after { content:''; flex:1; height:1px; background:var(--border); }

.result-found {
    background:linear-gradient(135deg,rgba(67,233,123,0.08),rgba(67,233,123,0.02));
    border:1px solid rgba(67,233,123,0.35); border-radius:20px;
    padding:2rem; text-align:center; position:relative; overflow:hidden;
}
.result-similar {
    background:linear-gradient(135deg,rgba(247,183,49,0.08),rgba(247,183,49,0.02));
    border:1px solid rgba(247,183,49,0.30); border-radius:20px;
    padding:1.75rem;
}
.result-not-found {
    background:linear-gradient(135deg,rgba(252,92,101,0.08),rgba(252,92,101,0.02));
    border:1px solid rgba(252,92,101,0.30); border-radius:20px;
    padding:2rem; text-align:center;
}

.song-match-icon   { font-size:2.5rem; margin-bottom:0.5rem; }
.match-label       { font-size:0.78rem; letter-spacing:3px; text-transform:uppercase; color:var(--success); margin-bottom:0.5rem; font-weight:500; }
.song-title-large  { font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:var(--text); line-height:1.1; margin:0.25rem 0; }
.confidence-pill   {
    display:inline-block; background:rgba(67,233,123,0.18);
    border:1px solid rgba(67,233,123,0.45); border-radius:100px;
    padding:0.3rem 1.1rem; font-size:0.88rem; color:var(--success); font-weight:600; margin-top:0.75rem;
}

.not-found-icon  { font-size:3rem; margin-bottom:0.5rem; }
.not-found-title { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#fc8b90; margin-bottom:0.4rem; }
.not-found-sub   { color:var(--muted); font-size:0.9rem; }

.similar-header {
    font-family:'Syne',sans-serif; font-size:0.82rem; font-weight:700;
    letter-spacing:2px; text-transform:uppercase; color:var(--warning); margin-bottom:1rem;
}
.similar-song-row {
    display:flex; align-items:center; gap:1rem;
    padding:0.85rem 0; border-bottom:1px solid var(--border);
}
.similar-song-row:last-child { border-bottom:none; }
.song-rank        { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:var(--muted); width:2rem; text-align:center; flex-shrink:0; }
.similar-song-name { font-weight:500; font-size:1rem; color:var(--text); flex:1; }
.confidence-bar-wrap { width:110px; background:var(--border); border-radius:100px; height:6px; overflow:hidden; flex-shrink:0; }
.confidence-bar-fill { height:100%; border-radius:100px; background:linear-gradient(90deg,#6c63ff,#ff6584); }
.conf-pct { font-size:0.8rem; color:var(--muted); width:4rem; text-align:right; flex-shrink:0; }

.tip-box     { background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.25); border-radius:14px; padding:1.1rem 1.4rem; margin-top:1.2rem; }
.tip-title   { font-family:'Syne',sans-serif; font-size:0.78rem; letter-spacing:2px; text-transform:uppercase; color:var(--accent); margin-bottom:0.6rem; }
.tip-item    { font-size:0.85rem; color:var(--muted); margin-bottom:0.3rem; }
.tip-item::before { content:'→ '; color:var(--accent); }

.status-row { display:flex; align-items:center; gap:0.6rem; font-size:0.82rem; color:var(--muted); margin-bottom:0.4rem; }
.dot        { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dot-green  { background:var(--success); box-shadow:0 0 6px var(--success); }

.stTabs [data-baseweb="tab-list"]  { background:var(--card); border-radius:14px; padding:0.3rem; gap:0.2rem; border:1px solid var(--border); }
.stTabs [data-baseweb="tab"]       { border-radius:10px!important; font-family:'DM Sans',sans-serif!important; font-size:0.9rem!important; color:var(--muted)!important; padding:0.5rem 1.5rem!important; }
.stTabs [aria-selected="true"]     { background:rgba(108,99,255,0.2)!important; color:#a09aff!important; }
.stTabs [data-baseweb="tab-panel"] { padding:1.2rem 0 0 0!important; }

.stButton button { width:100%; padding:0.75rem 1.5rem; border-radius:14px; font-family:'DM Sans',sans-serif; font-size:0.95rem; font-weight:500; border:none; cursor:pointer; transition:all 0.2s ease; }
.stButton button[kind="primary"]       { background:linear-gradient(135deg,#6c63ff,#9b59b6)!important; color:white!important; box-shadow:0 4px 20px rgba(108,99,255,0.35)!important; }
.stButton button[kind="secondary"]     { background:var(--card)!important; color:var(--muted)!important; border:1px solid var(--border)!important; }

.waveform-visual { display:flex; align-items:center; justify-content:center; gap:3px; height:60px; margin:1rem 0; }
.waveform-bar    { width:4px; border-radius:4px; background:linear-gradient(to top,#6c63ff,#ff6584); animation:wave 1.2s ease-in-out infinite; }
.waveform-bar:nth-child(1)  { height:20px; animation-delay:0.0s; }
.waveform-bar:nth-child(2)  { height:35px; animation-delay:0.1s; }
.waveform-bar:nth-child(3)  { height:50px; animation-delay:0.2s; }
.waveform-bar:nth-child(4)  { height:40px; animation-delay:0.3s; }
.waveform-bar:nth-child(5)  { height:55px; animation-delay:0.4s; }
.waveform-bar:nth-child(6)  { height:45px; animation-delay:0.5s; }
.waveform-bar:nth-child(7)  { height:60px; animation-delay:0.6s; }
.waveform-bar:nth-child(8)  { height:45px; animation-delay:0.5s; }
.waveform-bar:nth-child(9)  { height:55px; animation-delay:0.4s; }
.waveform-bar:nth-child(10) { height:40px; animation-delay:0.3s; }
.waveform-bar:nth-child(11) { height:50px; animation-delay:0.2s; }
.waveform-bar:nth-child(12) { height:35px; animation-delay:0.1s; }
.waveform-bar:nth-child(13) { height:20px; animation-delay:0.0s; }
@keyframes wave {
    0%,100% { transform:scaleY(0.4); opacity:0.5; }
    50%      { transform:scaleY(1.0); opacity:1.0; }
}

.song-pill { display:inline-block; background:rgba(108,99,255,0.12); border:1px solid rgba(108,99,255,0.25); border-radius:100px; padding:0.25rem 0.85rem; font-size:0.78rem; color:#a09aff; margin:0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
if "recorder_key" not in st.session_state:
    st.session_state.recorder_key = 0
if "result" not in st.session_state:
    st.session_state.result = None
if "reference_features_loaded" not in st.session_state:
    st.session_state.reference_features_loaded = False
if "reference_features" not in st.session_state:
    st.session_state.reference_features = None


def reset_recorder():
    st.session_state.recorder_key += 1
    st.session_state.result = None


def save_uploaded_file_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def save_recording_temp(audio_segment) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name
    audio_segment.export(tmp_path, format="wav")
    return tmp_path


def score_to_strength(top_score: float, second_score: float | None) -> float:
    base = float(np.clip(1.0 - top_score, 0.0, 1.0))
    if second_score is None:
        return base
    gap = max(0.0, second_score - top_score)
    gap_bonus = float(np.clip(gap / 0.25, 0.0, 1.0))
    strength = 0.75 * base + 0.25 * gap_bonus
    return float(np.clip(strength, 0.0, 1.0))


@st.cache_resource(show_spinner=False)
def load_reference_library():
    refs = discover_references()
    if not refs:
        return None, None, "references_missing"
    song_titles = sorted(ref.title_display for ref in refs.values())
    return refs, song_titles, "ok"


@st.cache_resource(show_spinner=False)
def build_reference_features():
    refs = discover_references()
    if not refs:
        return None

    built = {}
    for key, ref in refs.items():
        built[key] = safe_extract_reference_features(ref, use_cache=True)
    return built


def ensure_reference_features_loaded() -> bool:
    if st.session_state.reference_features_loaded and st.session_state.reference_features is not None:
        return True

    with st.spinner("Loading cached vocal reference features..."):
        ref_features = build_reference_features()

    if ref_features is None:
        return False

    st.session_state.reference_features = ref_features
    st.session_state.reference_features_loaded = True
    return True


def retrieve_audio(audio_path: str, top_k: int = TOP_K_DISPLAY) -> dict:
    try:
        if not ensure_reference_features_loaded():
            return {
                "status": "error",
                "error_msg": "Reference features could not be loaded.",
            }

        reference_features = st.session_state.reference_features

        audio, sr = librosa.load(audio_path, sr=SR, mono=True)

        if len(audio) < sr * 2.5:
            return {
                "status": "none",
                "top_conf": 0.0,
                "reason": "short",
            }

        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < 0.01:
            return {
                "status": "none",
                "top_conf": 0.0,
                "reason": "silent",
            }

        q_pitch, q_chroma = safe_extract_query_features(Path(audio_path))
        q_chroma_small = downsample_chroma(q_chroma, factor=4)

        # Stage 1: coarse shortlist
        coarse_rows = []
        for ref_key, feat in reference_features.items():
            coarse_score = np.inf
            if feat.vocals_chroma is not None:
                try:
                    coarse_score = dtw_chroma_distance(
                        q_chroma_small,
                        downsample_chroma(feat.vocals_chroma, factor=4),
                    )
                except Exception:
                    coarse_score = np.inf

            coarse_rows.append(
                {
                    "candidate_title_key": ref_key,
                    "candidate_title_display": feat.title_display,
                    "coarse_score": coarse_score,
                }
            )

        coarse_df = pd.DataFrame(coarse_rows)
        coarse_df = coarse_df.sort_values(
            ["coarse_score", "candidate_title_display"],
            ascending=[True, True],
        )
        shortlist_keys = coarse_df.head(SHORTLIST_SIZE)["candidate_title_key"].tolist()

        # Stage 2: fine ranking on shortlist
        fine_rows = []
        for ref_key in shortlist_keys:
            feat = reference_features[ref_key]

            row = {
                "candidate_title_display": feat.title_display,
                "candidate_title_key": ref_key,
                "vocal_pitch_dist": np.nan,
                "vocal_chroma_dist": np.nan,
            }

            if feat.vocals_pitch is not None:
                try:
                    row["vocal_pitch_dist"] = subseq_dtw_distance(q_pitch, feat.vocals_pitch)
                except Exception:
                    pass

            if feat.vocals_chroma is not None:
                try:
                    row["vocal_chroma_dist"] = dtw_chroma_distance(q_chroma, feat.vocals_chroma)
                except Exception:
                    pass

            fine_rows.append(row)

        cand_df = pd.DataFrame(fine_rows)

        if cand_df.empty:
            return {
                "status": "none",
                "top_conf": 0.0,
                "reason": "features",
            }

        score_cols = ["vocal_pitch_dist", "vocal_chroma_dist"]
        for col in score_cols:
            values = cand_df[col].to_numpy(dtype=np.float32)
            valid_mask = np.isfinite(values)
            norm = np.ones_like(values, dtype=np.float32)
            if valid_mask.any():
                norm_valid = minmax_normalize(values[valid_mask])
                norm[valid_mask] = norm_valid
            cand_df[col + "_norm"] = norm

        fused_scores = []
        for _, row in cand_df.iterrows():
            available = {}

            if pd.notna(row["vocal_pitch_dist"]):
                available["vocal_pitch_dist"] = VOCAL_ONLY_WEIGHTS["vocal_pitch"]
            if pd.notna(row["vocal_chroma_dist"]):
                available["vocal_chroma_dist"] = VOCAL_ONLY_WEIGHTS["vocal_chroma"]

            if not available:
                fused_scores.append(np.inf)
                continue

            weight_sum = sum(available.values())
            score = 0.0
            for col_name, weight in available.items():
                score += (weight / weight_sum) * float(row[col_name + "_norm"])
            fused_scores.append(score)

        cand_df["fused_score"] = fused_scores
        cand_df = cand_df.sort_values(
            ["fused_score", "candidate_title_display"],
            ascending=[True, True],
        ).reset_index(drop=True)

        top_score = float(cand_df.iloc[0]["fused_score"])
        second_score = float(cand_df.iloc[1]["fused_score"]) if len(cand_df) > 1 else None
        top_song = str(cand_df.iloc[0]["candidate_title_display"])
        top_strength = score_to_strength(top_score, second_score)

        results = []
        ordered = cand_df.head(top_k).reset_index(drop=True)
        for idx, row in ordered.iterrows():
            score = float(row["fused_score"])
            strength = score_to_strength(score, second_score if idx == 0 else None)
            results.append((str(row["candidate_title_display"]), strength, score))

        if not np.isfinite(top_score):
            return {
                "status": "none",
                "top_conf": 0.0,
                "reason": "low_confidence",
            }

        gap = (second_score - top_score) if second_score is not None else 1.0

        if top_score <= EXACT_SCORE_THRESHOLD and gap >= EXACT_GAP_THRESHOLD:
            return {
                "status": "exact",
                "top_song": top_song,
                "top_conf": top_strength,
                "similar": [(name, strength) for name, strength, _ in results[1:4]],
                "audio": audio,
                "sr": sr,
            }

        return {
            "status": "similar",
            "top_song": top_song,
            "top_conf": top_strength,
            "similar": [(name, strength) for name, strength, _ in results[:3]],
            "audio": audio,
            "sr": sr,
        }

    except Exception as exc:
        return {"status": "error", "error_msg": str(exc)}


def render_spectrogram(result: dict):
    audio = result.get("audio")
    sr = result.get("sr")
    if audio is None or sr is None:
        return

    with st.expander("🔬 View audio analysis"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), facecolor="#12121a")
        fig.patch.set_facecolor("#12121a")

        ax1 = axes[0]
        ax1.set_facecolor("#12121a")
        times = np.linspace(0, len(audio) / sr, len(audio))
        ax1.fill_between(times, audio, alpha=0.7, color="#6c63ff", linewidth=0)
        ax1.set_title("Waveform", color="#a09aff", fontsize=10, pad=8)
        ax1.set_xlabel("Time (s)", color="#7a7a9a", fontsize=8)
        ax1.tick_params(colors="#7a7a9a", labelsize=7)
        for sp in ax1.spines.values():
            sp.set_edgecolor("#2a2a3d")

        ax2 = axes[1]
        ax2.set_facecolor("#12121a")
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax2, cmap="magma")
        ax2.set_title("Mel Spectrogram", color="#a09aff", fontsize=10, pad=8)
        ax2.set_xlabel("Time (s)", color="#7a7a9a", fontsize=8)
        ax2.set_ylabel("")
        ax2.tick_params(colors="#7a7a9a", labelsize=7)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#2a2a3d")

        plt.tight_layout(pad=1.5)
        st.pyplot(fig)
        plt.close(fig)


def render_result_found(result: dict):
    name = result["top_song"]
    pct = f"{result['top_conf'] * 100:.0f}%"

    st.markdown(f"""
    <div class="result-found">
      <div class="song-match-icon">🎯</div>
      <div class="match-label">BEST MATCH</div>
      <div class="song-title-large">{name}</div>
      <span class="confidence-pill">Match strength {pct}</span>
    </div>
    """, unsafe_allow_html=True)

    if result.get("similar"):
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("See other close matches"):
            for i, (s_name, s_conf) in enumerate(result["similar"], 2):
                bar_w = int(s_conf * 100)
                st.markdown(f"""
                <div class="similar-song-row">
                  <div class="song-rank">#{i}</div>
                  <div class="similar-song-name">{s_name}</div>
                  <div class="confidence-bar-wrap">
                    <div class="confidence-bar-fill" style="width:{bar_w}%"></div>
                  </div>
                  <div class="conf-pct">{s_conf*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

    render_spectrogram(result)


def render_result_similar(result: dict):
    st.markdown("""
    <div class="result-similar">
      <div class="similar-header">🤔 &nbsp; Top matching songs</div>
      <p style="color:#b0b0c0; font-size:0.88rem; margin-bottom:1.2rem;">
        The melody is close to several songs. These are the strongest candidates.
      </p>
    """, unsafe_allow_html=True)

    for i, (s_name, s_conf) in enumerate(result.get("similar", []), 1):
        bar_w = int(s_conf * 100)
        st.markdown(f"""
        <div class="similar-song-row">
          <div class="song-rank">#{i}</div>
          <div class="similar-song-name">{s_name}</div>
          <div class="confidence-bar-wrap">
            <div class="confidence-bar-fill" style="width:{bar_w}%"></div>
          </div>
          <div class="conf-pct">{s_conf*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    render_spectrogram(result)


def render_result_none(result: dict):
    reason = result.get("reason", "low_confidence")

    if reason == "silent":
        subtitle = "No audio was detected. Make sure your microphone is working and try again."
    elif reason == "short":
        subtitle = "The recording was too short. Please hum for at least 3–5 seconds."
    elif reason == "features":
        subtitle = "Could not extract melody features. Try humming louder and more clearly."
    else:
        subtitle = "The melody did not match the song database confidently enough."

    st.markdown(f"""
    <div class="result-not-found">
      <div class="not-found-icon">🎵</div>
      <div class="not-found-title">No Strong Match</div>
      <div class="not-found-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tip-box">
      <div class="tip-title">💡 Tips to improve recognition</div>
      <div class="tip-item">Hum the chorus — it is the most recognisable part</div>
      <div class="tip-item">Hum continuously for at least 5 seconds</div>
      <div class="tip-item">Reduce background noise around you</div>
      <div class="tip-item">Hold the melody steadily instead of changing speed too much</div>
      <div class="tip-item">Try another take if the first one is unclear</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Load reference metadata only (fast) ──────────────────────────────────────
references, song_titles, load_status = load_reference_library()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <span class="hero-icon">🎵</span>
  <h1 class="hero-title">Hum2Tune</h1>
  <p class="hero-subtitle">
    Hum a melody — or play a song — and the retrieval engine will find the closest matches.
  </p>
  <span class="hero-badge">FYP Demo · Vocal-only Retrieval</span>
</div>
""", unsafe_allow_html=True)

if load_status == "references_missing":
    st.error("⚠️ Reference library not found. Make sure demucs vocal references and original songs exist.")
    st.stop()
else:
    n_songs = len(song_titles) if song_titles else 0
    st.markdown(f"""
    <div style="display:flex;gap:1.5rem;justify-content:center;margin-bottom:1.5rem;flex-wrap:wrap;">
      <div class="status-row"><div class="dot dot-green"></div> {RETRIEVAL_MODE_LABEL}</div>
      <div class="status-row"><div class="dot dot-green"></div> {n_songs} songs in database</div>
      <div class="status-row"><div class="dot dot-green"></div> Top-3 suggestions enabled</div>
      <div class="status-row"><div class="dot dot-green"></div> Demo shortlist: {SHORTLIST_SIZE}</div>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🎙️  Record yourself humming", "📂  Upload an audio file"])

audio_to_process = None
run_prediction = False

with tab1:
    st.markdown("""
    <div class="card">
      <div class="section-label">🔴 live recording</div>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:1rem;">
        Press <strong style="color:var(--text)">Record</strong>, hum the melody
        for at least 5 seconds, then press <strong style="color:var(--text)">Stop</strong>.
        The app will return the closest matching songs.
      </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="waveform-visual">
      <div class="waveform-bar"></div><div class="waveform-bar"></div>
      <div class="waveform-bar"></div><div class="waveform-bar"></div>
      <div class="waveform-bar"></div><div class="waveform-bar"></div>
      <div class="waveform-bar"></div><div class="waveform-bar"></div>
      <div class="waveform-bar"></div><div class="waveform-bar"></div>
      <div class="waveform-bar"></div><div class="waveform-bar"></div>
      <div class="waveform-bar"></div>
    </div>
    """, unsafe_allow_html=True)

    audio = audiorecorder(
        start_prompt="🔴  Start Recording",
        stop_prompt="⬛  Stop",
        key=f"rec_{st.session_state.recorder_key}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if len(audio) > 0:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">▶ review your recording</div>', unsafe_allow_html=True)
        st.audio(audio.export().read())

        col1, col2 = st.columns([3, 2])
        with col1:
            if st.button("🔍  Find Matching Songs", type="primary", key="btn_rec_go"):
                audio_to_process = save_recording_temp(audio)
                run_prediction = True
        with col2:
            if st.button("🗑️  Record Again", key="btn_rec_reset"):
                reset_recorder()
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="card">
      <div class="section-label">📂 upload audio</div>
      <p style="color:var(--muted);font-size:0.9rem;margin-bottom:1rem;">
        Upload a <strong style="color:var(--text)">.wav, .mp3, or .m4a</strong> file —
        humming or song audio. The app will rank the closest song matches.
      </p>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose audio file",
        type=["wav", "mp3", "m4a"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        st.audio(uploaded_file)

        if st.button("🔍  Find Matching Songs", type="primary", key="btn_upload_go"):
            audio_to_process = save_uploaded_file_temp(uploaded_file)
            run_prediction = True

    st.markdown("</div>", unsafe_allow_html=True)

if run_prediction and audio_to_process:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.spinner("Analysing melody patterns…"):
        time.sleep(0.15)
        st.session_state.result = retrieve_audio(audio_to_process)

    try:
        os.remove(audio_to_process)
    except OSError:
        pass

if st.session_state.result is not None:
    result = st.session_state.result
    st.markdown("<br>", unsafe_allow_html=True)

    if result["status"] == "exact":
        render_result_found(result)
    elif result["status"] == "similar":
        render_result_similar(result)
    elif result["status"] == "none":
        render_result_none(result)
    elif result["status"] == "error":
        st.markdown(f"""
        <div class="result-not-found">
          <div class="not-found-icon">⚠️</div>
          <div class="not-found-title">Something went wrong</div>
          <div class="not-found-sub">{result.get("error_msg", "Unknown error")}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↩  Try Another Song", key="btn_try_again"):
        reset_recorder()
        st.rerun()

st.markdown("<br><br>", unsafe_allow_html=True)
if song_titles:
    with st.expander(f"🎶  Song Database  ({len(song_titles)} songs available)"):
        st.markdown("""
        <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        The retrieval engine compares your melody against the following reference songs.
        </p>
        """, unsafe_allow_html=True)
        pills = "".join(
            f'<span class="song-pill">{name}</span>'
            for name in song_titles
        )
        st.markdown(pills, unsafe_allow_html=True)