# PS3DT — Audio Deepfake Detection

A Streamlit frontend for the PS3DT model: a 12-layer transformer that classifies
a speech clip as bonafide (real) or spoof (synthetic/replayed), trained on
ASVspoof 2019 (Logical Access).

## What's inside

```
app.py                     Streamlit frontend (UI, layout, plots)
backend/
  model.py                 PS3DT architecture (matches the training notebook)
  audio_processing.py      waveform -> log-mel -> patches pipeline
  inference.py              model/checkpoint loading + prediction
assets/style.css           Dark-navy / light-blue theme
.streamlit/config.toml     Streamlit theme + server settings
checkpoints/                put best_param.pth (or finetune.pth) here
requirements.txt
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Add your trained model

The weights file (`best_param.pth`, ~338MB) is too big for a normal GitHub
push (GitHub hard-blocks files over 100MB), so it's never committed to the
repo. Instead:

- **Local runs**: just drop the file into `checkpoints/best_param.pth`
  directly — the app finds it on disk and skips downloading anything.
- **Deployed runs**: the app downloads it automatically at startup from a
  URL you configure (see the deployment guide below). See
  `backend/download.py` for how this works.

`checkpoints/finetune.pth` (the wrapped `{"model_state_dict": ...}` training
checkpoint format) also works, either way.

Without a checkpoint present or downloadable, the app still runs end-to-end
with randomly initialised weights, and shows a "Demo weights" banner so
that's obvious.

## Deploy to Streamlit Community Cloud

### 1. Host the weights file somewhere with a direct download URL

Hugging Face Hub is the easiest free option for a ~338MB file:

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Model** (top right → New → Model), give it a name (e.g.
   `ps3dt-weights`), and set visibility to **Public** (Private repos need
   an access token wired into your app, which adds an extra step).
3. On the model page, click **Files and versions → Add file → Upload
   files**, and upload `best_param.pth`. No Git LFS setup needed — the
   web uploader handles large files on its own.
4. Once uploaded, your direct download URL is:
   ```
   https://huggingface.co/<your-username>/ps3dt-weights/resolve/main/best_param.pth
   ```
   Open it in a browser to confirm it triggers a file download.

(Any host that gives a direct, no-login URL works the same way — e.g. a
GitHub Release asset, Dropbox with `?dl=1`, or your own storage bucket.)

### 2. Push the code (without the weights) to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

`checkpoints/*.pth` is already in `.gitignore`, so the weights file won't
be included even if it's sitting in that folder locally.

### 3. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, pick your repo/branch, and set the main file to `app.py`.
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```toml
   CHECKPOINT_URL = "https://huggingface.co/<your-username>/ps3dt-weights/resolve/main/best_param.pth"
   ```
4. Click **Deploy**. First build takes a few minutes (PyTorch is a big
   dependency, plus the one-time weights download) — subsequent restarts
   are faster since the file stays cached on that container until it's
   redeployed or goes to sleep from inactivity.

If you ever need to update the secret after deploying: app menu (⋮) →
**Settings → Secrets → Edit**, then reboot the app from the same menu.

## Notes

- Audio is resampled to 16 kHz mono, then tiled or truncated to a fixed
  5.12 second window (81,920 samples), matching how the model was trained.
- The sidebar's decision threshold only changes how a probability is
  labeled SPOOF/BONAFIDE in the UI — it does not affect the model's raw
  output probabilities.
- This is a research/demo tool. It should support human review, not replace
  it, for any high-stakes authenticity decision.
