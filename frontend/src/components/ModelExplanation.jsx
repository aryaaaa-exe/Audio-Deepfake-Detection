import "./ModelExplanation.css";

export default function ModelExplanation() {
  return (
    <section className="model-section">

      <h2>How the Model Works</h2>

      <p className="model-subtitle">
        The uploaded audio passes through multiple processing stages before
        being classified as <strong>Bonafide</strong> or <strong>Spoof</strong>.
      </p>

      <div className="pipeline-container">

        <div className="pipeline-card">
          <div className="pipeline-icon">🎵</div>
          <h3>Audio Preprocessing</h3>
          <p>
            The uploaded audio is resampled to <b>16 kHz</b> and adjusted to a
            fixed duration by repeating or truncating the waveform.
          </p>
        </div>

        <div className="arrow">→</div>

        <div className="pipeline-card">
          <div className="pipeline-icon">📊</div>
          <h3>Mel Spectrogram</h3>
          <p>
            The waveform is converted into a Mel Spectrogram, providing a
            time-frequency representation that highlights speech characteristics.
          </p>
        </div>

        <div className="arrow">→</div>

        <div className="pipeline-card">
          <div className="pipeline-icon">🧩</div>
          <h3>Patch Embedding</h3>
          <p>
            The <b>80 × 512</b> spectrogram is divided into
            <b> 160 non-overlapping patches (16 × 16)</b>. Processing smaller
            regions helps the Transformer learn local acoustic patterns while
            also understanding relationships across the entire spectrogram.
          </p>
        </div>

        <div className="arrow">→</div>

        <div className="pipeline-card">
          <div className="pipeline-icon">🧠</div>
          <h3>Transformer Encoder</h3>
          <p>
            Self-attention compares all patches simultaneously, enabling the
            model to detect subtle artifacts commonly present in AI-generated
            speech.
          </p>
        </div>

        <div className="arrow">→</div>

        <div className="pipeline-card">
          <div className="pipeline-icon">✅</div>
          <h3>Classification</h3>
          <p>
            The extracted features are combined to predict whether the audio is
            <b> Bonafide</b> or <b>Spoof</b>, along with a confidence score.
          </p>
        </div>

      </div>

    </section>
  );
}