"""
Inference layer that sits between the Streamlit frontend and the PS3DT model.

Handles: device selection, checkpoint loading (with a graceful fallback to
randomly-initialised weights so the app still runs end-to-end before a
trained checkpoint is added), and turning raw audio bytes into a result
dictionary the UI can render directly.
"""
from huggingface_hub import hf_hub_download
import os
import time
from dataclasses import dataclass, field

import torch

from .model import PS3DT
from .audio_processing import audio_to_model_input
#from .download import ensure_checkpoint

# CHECKPOINT_CANDIDATES = [
#     os.path.join("checkpoints", "best_param.pth"),
#     os.path.join("checkpoints", "finetune.pth"),
# ]


@dataclass
class PredictionResult:
    label: str                 # "BONAFIDE" or "SPOOF"
    prob_real: float           # 0-100
    prob_fake: float           # 0-100
    waveform: "np.ndarray"
    mel: "np.ndarray"
    sample_rate: int
    inference_ms: float
    demo_mode: bool
    checkpoint_name: str = field(default="")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def find_checkpoint():
#     for candidate in CHECKPOINT_CANDIDATES:
#         if os.path.exists(candidate):
#             return candidate
#     return None


def load_model():
    device = get_device()
    model = PS3DT().to(device)

    # checkpoint_path = find_checkpoint()
    # if checkpoint_path is None:
    #     # Not found locally — try pulling it from a configured remote URL
    #     # (see backend/download.py). No-ops if no URL is configured.
    #     checkpoint_path = ensure_checkpoint()

    checkpoint_path = hf_hub_download(
        repo_id="aryaaaa-exe/ps3dt-audio-deepfake",
        filename="best_param.pth"
    )
    checkpoint_name = os.path.basename(checkpoint_path)
    # demo_mode = checkpoint_path is None
    # checkpoint_name = ""

    # if checkpoint_path is not None:
    #     checkpoint_name = os.path.basename(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device)
        # Support both a raw state_dict and a training-checkpoint dict
        # that wraps it under "model_state_dict".
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)

    model.eval()
    return model, device, False, checkpoint_name #demo_mode


def predict(model, device, demo_mode, checkpoint_name, audio_source) -> PredictionResult:
    waveform, mel, patches = audio_to_model_input(audio_source)
    patches = patches.to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(patches)
        probs = torch.softmax(logits, dim=1)
    elapsed_ms = (time.perf_counter() - start) * 1000

    prob_real = probs[0][0].item() * 100
    prob_fake = probs[0][1].item() * 100
    label = "SPOOF" if prob_fake > prob_real else "BONAFIDE"

    return PredictionResult(
        label=label,
        prob_real=prob_real,
        prob_fake=prob_fake,
        waveform=waveform,
        mel=mel,
        sample_rate=16000,
        inference_ms=elapsed_ms,
        demo_mode=demo_mode,
        checkpoint_name=checkpoint_name,
    )
