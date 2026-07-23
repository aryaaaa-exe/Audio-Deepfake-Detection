import librosa
import numpy as np

#Function For Feature Extraction
def extract_mel_spectrogram(waveform,sr):
  mel_spectrogram=librosa.feature.melspectrogram(
      y=waveform,
      sr=sr,
      n_fft=400, #the paper says "25 ms window" so 25ms=0.025sec. So, 0.025*sr=400
      hop_length=160, #the paper says "10 ms stride" so 10ms=0.01sec. So, 0.01*sr=160
      n_mels=80, #80-bin Log-Mel Spectrogram
      window='hann'
  )

  #Convert power spectrogram to log scale (dB)
  log_mel=librosa.power_to_db(
      mel_spectrogram,
      ref=np.max
  )

  #Keep only the first 512 time frames
  log_mel=log_mel[:, :512]
  #without this line we were getting shape as (80*513) due to centre=True padding of librosa


  return log_mel