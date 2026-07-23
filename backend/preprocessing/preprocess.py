import librosa
import numpy as np

target_sr=16000
target_duration=5.12
target_length=int(target_sr*target_duration)

#loading the audio
def load_audio(audio_path):
    waveform,sampling_rate=librosa.load(
        audio_path,
        sr=target_sr,
        mono=True
    )

    return waveform

#preprocess audio
def preprocess_audio(waveform,target_length):
  current_length=len(waveform)
  if current_length==0: #handling an edge case
    raise ValueError("Encountered an empty audio waveform.")
  if current_length<target_length:
    repeat_factor=int(np.ceil(target_length/current_length)) #the number of time the audio will be repeated
    waveform=np.tile(waveform,repeat_factor) #construct a new array by repeating an existing array a specified number of times.
  waveform=waveform[:target_length] #truncate(similar to slicing)
  return waveform

