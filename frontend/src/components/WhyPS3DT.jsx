import "./WhyPS3DT.css";

export default function WhyPS3DT() {
  return (
    <section className="why-section">

      <h2>Why PS3DT?</h2>

      <p className="why-subtitle">
        PS3DT (Patch-based Spectrogram Swin/Transformer Deepfake Detector)
        analyzes audio more effectively than conventional CNN-based models by
        learning relationships across the entire spectrogram.
      </p>

      <div className="comparison-table">

        <div className="table-header">Traditional CNN</div>
        <div className="table-header ps3dt-header">PS3DT Transformer</div>

        <div className="table-cell">
          Focuses mainly on local features.
        </div>
        <div className="table-cell highlight">
          Learns both local and global relationships using self-attention.
        </div>

        <div className="table-cell">
          Limited receptive field.
        </div>
        <div className="table-cell highlight">
          Every patch can attend to every other patch.
        </div>

        <div className="table-cell">
          Requires deep stacking to capture long-range dependencies.
        </div>
        <div className="table-cell highlight">
          Captures long-range dependencies naturally through attention.
        </div>

        <div className="table-cell">
          Less effective at detecting subtle synthetic artifacts.
        </div>
        <div className="table-cell highlight">
          Better at identifying inconsistencies introduced by AI-generated speech.
        </div>

        <div className="table-cell">
          Processes the spectrogram as a whole feature map.
        </div>
        <div className="table-cell highlight">
          Processes 160 spectrogram patches as Transformer tokens.
        </div>

      </div>

    </section>
  );
}