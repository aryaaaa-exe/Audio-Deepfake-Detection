import { useRef, useState } from "react";
import { predictAudio } from "C:/Users/Asus/Desktop/Coding/Audio-Deepfake-Detection/frontend/src/api/predict";
import ProcessingPipeline from "C:/Users/Asus/Desktop/Coding/Audio-Deepfake-Detection/frontend/src/components/ProcessingPipeline";
import AnimatedCard from "../animations/AnimatedCard";
import { useEffect } from "react";

export default function UploadBox() {

  const [selectedFile, setSelectedFile] = useState(null);
  const [audioURL, setAudioURL] = useState("");
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(-1);
  const [spectrogram, setSpectrogram] = useState("");
  const canvasRef = useRef(null);

  const fileInputRef = useRef(null);

  function handleButtonClick() {
    fileInputRef.current.click();
  }

  function handleFileChange(event) {

    const file = event.target.files[0];

    if (!file) return;

    setSelectedFile(file);

    const url = URL.createObjectURL(file);

    setAudioURL(url);
  }

  function removeFile() {

    setSelectedFile(null);

    setAudioURL("");

    fileInputRef.current.value = "";

  }
  useEffect(() => {

    if (!selectedFile) return;

    const reader = new FileReader();

    reader.onload = async (e) => {

        const audioContext =
            new AudioContext();

        const buffer =
            await audioContext.decodeAudioData(
                e.target.result
            );

        drawWaveform(buffer);

    };

    reader.readAsArrayBuffer(selectedFile);

}, [selectedFile]);

function drawWaveform(audioBuffer){

const canvas=canvasRef.current;

if(!canvas)return;

const ctx=canvas.getContext("2d");

const width=canvas.width;

const height=canvas.height;

ctx.clearRect(0,0,width,height);

ctx.strokeStyle="#3B82F6";

ctx.lineWidth=2;

ctx.beginPath();

const data=audioBuffer.getChannelData(0);

const step=Math.ceil(data.length/width);

const amp=height/2;

for(let i=0;i<width;i++){

let min=1;

let max=-1;

for(let j=0;j<step;j++){

const datum=data[(i*step)+j];

if(datum<min)min=datum;

if(datum>max)max=datum;

}

ctx.moveTo(i,(1+min)*amp);

ctx.lineTo(i,(1+max)*amp);

}

ctx.stroke();

}

  async function analyzeAudio() {

  if (!selectedFile) return;

  setPrediction("");
  setConfidence(null);

  setCurrentStep(0);

  for (let i = 0; i < 5; i++) {

    setCurrentStep(i);

    await new Promise(resolve =>
      setTimeout(resolve, 700)
    );

  }

  try {

    setLoading(true);

    const result =
      await predictAudio(selectedFile);

    setPrediction(result.prediction);

    setConfidence(result.confidence);
    setSpectrogram(result.spectrogram);

  }
  catch(error){

    console.error(error);

    alert("Backend connection failed.");

  }
  finally{

    setLoading(false);

    setCurrentStep(-1);

  }

}

  return (
<AnimatedCard>
    <div className="upload-box">

      <div className="upload-icon">🎵</div>

      <h2>Upload Audio File</h2>

      {!selectedFile && (
        <>
          <p>Drag & Drop your audio here</p>

          <button onClick={handleButtonClick}>
            Choose Audio
          </button>
        </>
      )}

      <input
        type="file"
        accept=".wav,.mp3,.flac"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {selectedFile && (
        <>

          <p className="filename">

            {selectedFile.name}

          </p>

          <audio
            controls
            src={audioURL}
            className="audio-player"
          />
          <canvas

            ref={canvasRef}

            width={700}

            height={160}

            className="waveform-canvas"

        />

          <div className="button-group">

    <button
        className="analyze-btn"
        onClick={analyzeAudio}
    >
        Analyze Audio
    </button>

    <button
        className="remove-btn"
        onClick={removeFile}
    >
        Remove
    </button>

</div>

{/* {loading && (
    <ProcessingPipeline
        currentStep={currentStep}
        visible={true}
    />
)} */}
<ProcessingPipeline
    spectrogram={spectrogram}
/>

{prediction && (

    <div className="result-card">

        <h2>{prediction}</h2>

        <p>

            Confidence: {confidence.toFixed(2)}%

        </p>

    </div>

)}
        </>
      )}

      <small>

        Supported: .wav • .flac • .mp3

      </small>

    </div>
    </AnimatedCard>

  );
}