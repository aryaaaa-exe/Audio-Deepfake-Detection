from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import os
import matplotlib.pyplot as plt
import uuid

from inference import predict

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/")
def home():
    return "Audio Deepfake Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict_audio():

    # Check if an audio file was uploaded
    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file uploaded."
        }), 400

    file = request.files["audio"]

    # Check if filename is empty
    if file.filename == "":
        return jsonify({
            "error": "No file selected."
        }), 400

    # Save uploaded file
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(file_path)

    try:

        prediction, confidence, mel_filename = predict(file_path)

        return jsonify({

            "prediction": prediction,

            "confidence": round(confidence, 2),
            "spectrogram": url_for(
                "static",
                filename=mel_filename,
                _external=True
            )
            # "spectrogram": f"http://127.0.0.1:5000/static/{mel_filename}"

        })

    except Exception as e:

        return jsonify({

            "error": str(e)

        }), 500

    finally:
        # Delete uploaded file after prediction
        if os.path.exists(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)