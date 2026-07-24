"""
Audio preprocessing pipeline for PS3DT.

Mirrors the exact preprocessing / feature-extraction steps used in the
research notebook so that inference results are consistent with training:

    1. Resample to 16 kHz
    2. Pad (tile) or truncate to 5.12 seconds (81,920 samples)
    3. 80-bin log-mel spectrogram, 25 ms window / 10 ms hop
    4. Slice to 512 time frames
    5. Split into non-overlapping 16x16 patches, flatten to (160, 256)
"""

import numpy as np
import librosa
import torch

SAMPLE_RATE = 16000
DESIRED_SECONDS = 5.12
TARGET_LENGTH = int(DESIRED_SECONDS * SAMPLE_RATE)  # 81,920 samples
PATCH_SIZE = 16
N_MELS = 80
N_FFT = 400        # 25 ms window at 16 kHz
HOP_LENGTH = 160    # 10 ms stride at 16 kHz
N_FRAMES = 512


def load_waveform(path_or_buffer):
    """Load audio from a path or file-like object, resampled to 16 kHz mono."""
    waveform, sr = librosa.load(path_or_buffer, sr=SAMPLE_RATE, mono=True)
    return waveform, sr


def preprocess_audio(waveform: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    """Tile short clips up to length, truncate long clips down to length."""
    current_length = len(waveform)
    if current_length == 0:
        raise ValueError("Encountered an empty audio waveform.")
    if current_length < target_length:
        repeat_factor = int(np.ceil(target_length / current_length))
        waveform = np.tile(waveform, repeat_factor)
    return waveform[:target_length]


def extract_mel_spectrogram(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """80-bin log-mel spectrogram, cropped to the first 512 time frames."""
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window="hann",
    )
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel[:, :N_FRAMES]


def create_patches(mel_spectrogram: np.ndarray, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """Splits an (80, 512) spectrogram into 160 non-overlapping 16x16 patches."""
    patches = []
    for i in range(0, mel_spectrogram.shape[0], patch_size):
        for j in range(0, mel_spectrogram.shape[1], patch_size):
            patches.append(mel_spectrogram[i : i + patch_size, j : j + patch_size])
    patches = np.array(patches)
    return torch.tensor(patches, dtype=torch.float32)


def audio_to_model_input(path_or_buffer):
    """Full pipeline: raw audio -> (waveform, mel, patch tensor ready for the model).

    Returns:
        waveform (np.ndarray): preprocessed 5.12s waveform at 16kHz, for playback/plotting.
        mel (np.ndarray): (80, 512) log-mel spectrogram, for visualization.
        patches (torch.Tensor): (1, 160, 256) tensor, ready for PS3DT.forward().
    """
    waveform, sr = load_waveform(path_or_buffer)
    waveform = preprocess_audio(waveform, TARGET_LENGTH)
    mel = extract_mel_spectrogram(waveform, sr)
    patches = create_patches(mel, PATCH_SIZE)
    patches = patches.reshape(patches.shape[0], -1)  # (160, 16, 16) -> (160, 256)
    patches = patches.unsqueeze(0)  # add batch dimension -> (1, 160, 256)
    return waveform, mel, patches
