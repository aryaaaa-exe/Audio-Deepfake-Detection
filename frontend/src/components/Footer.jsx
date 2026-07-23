export default function Footer() {
  return (
    <footer className="footer">

      <div className="footer-content">

        <h2>Audio Deepfake Detection</h2>

        <p className="footer-description">
          An AI-powered system for detecting synthetic speech using
          <strong> PS3DT (Patch-based Spectrogram Transformer)</strong>.
        </p>

        <div className="footer-tech">

          <span>⚛️ React</span>
          <span>🐍 Flask</span>
          <span>🔥 PyTorch</span>
          <span>🎵 Librosa</span>

        </div>

        <div className="footer-line"></div>

        <p className="footer-bottom">
          © 2026 Audio Deepfake Detection • Built for Educational & Research Purposes
        </p>

      </div>

    </footer>
  );
}