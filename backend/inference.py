import torch
import matplotlib.pyplot as plt
import uuid
from huggingface_hub import hf_hub_download
import os

from model.ps3dt import PS3DT
from preprocessing.preprocess import (
    load_audio,
    preprocess_audio,
    target_length,
    target_sr
)
from preprocessing.feature_extraction import extract_mel_spectrogram
from preprocessing.patch_generator import (
    create_patches,
    flatten_patches
)

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
def load_model():
    global model

    if model is not None:
        return model

    model = PS3DT()

    MODEL_PATH = "weights/best_param.pth"

    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = hf_hub_download(
            repo_id="aryaaaa-exe/audio-deepfake-weights",
            filename="best_param.pth"
        )

    state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model

#Prediction Function
def predict(audio_path):
    model = load_model()
    # Load audio
    waveform = load_audio(audio_path)

    # Preprocess
    waveform = preprocess_audio(
        waveform,
        target_length
    )

    # Feature Extraction
    mel = extract_mel_spectrogram(
        waveform,
        target_sr
    )

    # Patch Creation
    patches = create_patches(mel)

    # Flatten
    patches = flatten_patches(patches)

    # Add Batch Dimension
    patches = patches.unsqueeze(0)

    patches = patches.to(device)

    # Prediction
    with torch.no_grad():

        logits = model(patches)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100

    if predicted_class == 0:
        label = "Bonafide"
    else:
        label = "Spoof"

    mel_filename = f"{uuid.uuid4()}.png"

    mel_path = os.path.join(
        "static",
        mel_filename
    )

    plt.figure(figsize=(10,4))

    plt.imshow(
        mel,
        origin="lower",
        aspect="auto",
        cmap="magma"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0
    )

    plt.savefig(
        mel_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    return label, confidence, mel_filename


#Test
if __name__ == "__main__":

    audio_path = "sample.flac"      # Replace with your test file

    label, confidence = predict(audio_path)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2f}%")