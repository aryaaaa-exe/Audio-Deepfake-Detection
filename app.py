import io
import random

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from backend.inference import load_model, predict

# --------------------------------------------------------------------------
# Page config + theme
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="PS3DT | Audio Deepfake Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

PLOTLY_TRANSPARENT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9fb3dc", family="Inter, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)


# --------------------------------------------------------------------------
# Cached model load
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading PS3DT model weights...")
def get_model():
    return load_model()


model, device, demo_mode, checkpoint_name = get_model()


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:10px;margin-bottom:2px;'>"
        "<span style='font-size:1.5rem;color:#45e6ff;'>◈</span>"
        "<span style='font-family:Space Grotesk,sans-serif;font-size:1.15rem;"
        "font-weight:700;color:#eaf2ff;'>PS3DT</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Patch-based Spectrogram Transformer for audio spoof detection")

    st.markdown("---")
    st.markdown("**Runtime status**")
    status_color = "#ff8a65" if demo_mode else "#49e0b3"
    status_text = "Demo weights (untrained)" if demo_mode else f"Checkpoint: {checkpoint_name}"
    st.markdown(
        f"<div class='ps-badge dot' style='border-color:{status_color}55;color:{status_color};'>"
        f"{status_text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<div class='mono' style='font-size:0.78rem;margin-top:10px;color:#5d719e;'>"
                f"device: {device.type}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Architecture**")
    st.markdown(
        """
        <div class='mono' style='font-size:0.8rem;line-height:2.0;color:#9fb3dc;'>
        encoder layers &nbsp;12<br>
        attention heads &nbsp;12<br>
        model dim &nbsp;768<br>
        patch size &nbsp;16 x 16<br>
        patches / clip &nbsp;160<br>
        input length &nbsp;5.12 s
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "Spoof probability above this value is flagged as SPOOF",
        min_value=5, max_value=95, value=50, step=1,
        label_visibility="collapsed",
    )
    st.caption(f"Flag as SPOOF when spoof probability > {threshold}%")

    st.markdown("---")
    st.caption("Trained on ASVspoof 2019 (Logical Access). Not intended as a sole authenticity guarantee for high-stakes decisions.")


# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
random.seed(7)
bars = "".join(
    f"<span style='height:{h}%;animation-delay:{d:.2f}s;'></span>"
    for h, d in [(random.randint(20, 100), random.uniform(0, 1.4)) for _ in range(46)]
)

hero_l, hero_r = st.columns([1.15, 1], gap="large")
with hero_l:
    st.markdown("<div class='ps-eyebrow'>Speech Deepfake Detection</div>", unsafe_allow_html=True)
    st.markdown(
        "<h1 class='ps-hero-title'>Tell real speech<br>from "
        "<span class='ps-gradient-text'>synthetic voice</span>.</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='ps-hero-sub'>PS3DT turns a raw waveform into a log-mel spectrogram, "
        "slices it into patches, and runs them through a 12-layer transformer encoder "
        "trained to separate bonafide human speech from synthetic and replayed audio. "
        "Upload a clip below to see it work.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='ps-badge dot' style='margin-right:8px;'>ASVspoof 2019 · LA</div>"
        "<div class='ps-badge dot'>Transformer Encoder</div>",
        unsafe_allow_html=True,
    )
with hero_r:
    st.markdown(f"<div class='ps-wave'>{bars}</div>", unsafe_allow_html=True)

st.markdown("<div style='height:34px;'></div>", unsafe_allow_html=True)

if demo_mode:
    st.markdown(
        "<div class='ps-banner'>Running with randomly initialised weights — no trained "
        "checkpoint was found at <span class='mono'>checkpoints/best_param.pth</span>. "
        "Predictions below are for pipeline demonstration only. Drop your trained "
        "checkpoint into that folder and reload to get real predictions.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------
tab_detect, tab_pipeline, tab_about = st.tabs(["Detect", "How it works", "About"])

# ---- DETECT ----
with tab_detect:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown("#### Upload audio")
        st.caption("WAV, FLAC, or MP3 — any length. It will be resampled to 16 kHz and fit to 5.12s.")
        uploaded = st.file_uploader(
            "Upload audio", type=["wav", "flac", "mp3", "ogg"], label_visibility="collapsed"
        )
        run = st.button("Analyze audio", use_container_width=True, disabled=uploaded is None)

        if uploaded is not None:
            st.audio(uploaded)

    with right:
        result_slot = st.container()

    if run and uploaded is not None:
        with st.spinner("Extracting features and running the encoder..."):
            audio_bytes = io.BytesIO(uploaded.getvalue())
            result = predict(model, device, demo_mode, checkpoint_name, audio_bytes)
        st.session_state["ps3dt_result"] = result

    result = st.session_state.get("ps3dt_result")

    with result_slot:
        if result is None:
            st.markdown(
                "<div class='ps-card' style='text-align:center;padding:64px 24px;'>"
                "<div style='font-size:2rem;color:#5d719e;'>◈</div>"
                "<p style='margin-top:10px;color:#5d719e;'>Upload a clip and press "
                "<b style='color:#9fb3dc;'>Analyze audio</b> to see the verdict, "
                "confidence, and spectrogram here.</p></div>",
                unsafe_allow_html=True,
            )
        else:
            is_spoof = result.prob_fake > threshold
            verdict_class = "fake" if is_spoof else "real"
            verdict_word = "LIKELY SPOOF" if is_spoof else "LIKELY BONAFIDE"
            confidence = result.prob_fake if is_spoof else result.prob_real

            st.markdown(
                f"""
                <div class='ps-verdict {verdict_class}'>
                    <div class='ps-eyebrow' style='margin-bottom:8px;'>Model verdict</div>
                    <p class='ps-verdict-label'>{verdict_word}</p>
                    <p class='ps-verdict-sub'>{confidence:.1f}% confidence at a {threshold}% threshold</p>
                    <div class='ps-gauge'>
                        <div class='ps-gauge-fill {verdict_class}' style='width:{confidence:.1f}%;'></div>
                    </div>
                    <div class='ps-readout'>
                        <div>BONAFIDE PROB.<b>{result.prob_real:.2f}%</b></div>
                        <div>SPOOF PROB.<b>{result.prob_fake:.2f}%</b></div>
                        <div>INFERENCE TIME<b>{result.inference_ms:.0f} ms</b></div>
                        <div>DEVICE<b>{device.type.upper()}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result is not None:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        wcol, mcol = st.columns(2, gap="large")

        with wcol:
            st.markdown("##### Waveform (preprocessed, 5.12s @ 16kHz)")
            t = np.linspace(0, len(result.waveform) / result.sample_rate, len(result.waveform))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t, y=result.waveform, mode="lines",
                line=dict(color="#45e6ff", width=1),
            ))
            fig.update_layout(height=280, xaxis_title="seconds", **PLOTLY_TRANSPARENT)
            fig.update_xaxes(gridcolor="rgba(139,178,255,0.08)")
            fig.update_yaxes(gridcolor="rgba(139,178,255,0.08)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with mcol:
            st.markdown("##### Log-mel spectrogram (80 x 512)")
            fig2 = go.Figure(data=go.Heatmap(
                z=result.mel, colorscale=[[0, "#050912"], [0.5, "#2151a8"], [1, "#45e6ff"]],
                showscale=False,
            ))
            fig2.update_layout(height=280, xaxis_title="frame", yaxis_title="mel bin", **PLOTLY_TRANSPARENT)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ---- HOW IT WORKS ----
with tab_pipeline:
    st.markdown("#### From waveform to verdict")
    st.caption("Every clip passes through five stages before a decision is made.")

    steps = [
        ("01", "Load & normalize", "Audio is resampled to 16kHz mono, then tiled or truncated to a fixed 5.12s (81,920 samples)."),
        ("02", "Log-mel spectrogram", "An 80-bin mel spectrogram is computed with a 25ms window and 10ms stride, then converted to log scale."),
        ("03", "Patch embedding", "The 80x512 spectrogram is split into 160 non-overlapping 16x16 patches, each flattened and linearly projected to 768 dims."),
        ("04", "Transformer encoder", "12 self-attention blocks (12 heads, MSM-MAE style zero key-bias) build context across all 160 patches."),
        ("05", "Classification head", "Patches are pooled per time-frame, averaged, and passed through a small MLP to output bonafide vs. spoof probabilities."),
    ]
    cols = st.columns(len(steps))
    for col, (idx, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div class='ps-pipe-step'>
                    <div class='ps-pipe-index'>{idx}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Architecture at a glance")
    s1, s2, s3, s4 = st.columns(4)
    for col, num, label in [
        (s1, "12", "Encoder layers"),
        (s2, "768", "Model dimension"),
        (s3, "160", "Patches / clip"),
        (s4, "12", "Attention heads"),
    ]:
        with col:
            st.markdown(
                f"<div class='ps-stat'><div class='ps-stat-num'>{num}</div>"
                f"<div class='ps-stat-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.85rem;'>The encoder is initialised from MSM-MAE pretrained weights "
        "(with decoder and patch-embedding weights excluded), then fine-tuned end-to-end on "
        "ASVspoof 2019 with a weighted sampler to correct for class imbalance between bonafide "
        "and spoof clips.</p>",
        unsafe_allow_html=True,
    )

# ---- ABOUT ----
with tab_about:
    st.markdown("#### About this project")
    st.markdown(
        """
        <div class='ps-card'>
        <p>PS3DT is a transformer-based classifier built to separate bonafide human speech
        from synthetic, converted, and replayed audio (commonly called audio spoofing or
        audio deepfakes). It treats a speech clip as an image-like grid of spectrogram
        patches and reuses ideas from masked spectrogram modeling to encode it, rather than
        processing the raw waveform directly.</p>
        <p style="margin-top:14px;">The model is trained and evaluated on the
        <b style="color:#eaf2ff;">ASVspoof 2019 Logical Access</b> dataset, using Equal Error
        Rate (EER) as the primary evaluation metric — the point on the ROC curve where the
        false-accept and false-reject rates are equal.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)
    st.markdown("#### Good to know")
    st.markdown(
        """
        <div class='ps-card'>
        <p>• Trained on English speech from ASVspoof 2019 — accuracy on other languages,
        accents, or newer generative TTS/voice-conversion systems is not guaranteed.</p>
        <p style="margin-top:10px;">• Clips are cropped or looped to exactly 5.12 seconds;
          very short or very long clips are adapted to fit this window.</p>
        <p style="margin-top:10px;">• This tool supports human review — it should not be the
          sole basis for high-stakes authenticity decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div class='ps-footer'>PS3DT · Patch-based Spectrogram Transformer for Audio Spoof Detection · "
    "Built on ASVspoof 2019</div>",
    unsafe_allow_html=True,
)
