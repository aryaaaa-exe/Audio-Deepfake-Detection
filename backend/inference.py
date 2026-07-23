import torch
import matplotlib.pyplot as plt
import uuid
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

#Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PS3DT()
state_dict = torch.load(
    "weights/best_param.pth",
    map_location=device
)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

#Prediction Function
def predict(audio_path):

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