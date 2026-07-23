from preprocessing.preprocess import load_audio, preprocess_audio, target_sr, target_length
from preprocessing.feature_extraction import extract_mel_spectrogram
from preprocessing.patch_generator import create_patches, flatten_patches

audio_path = "sample.flac"     # we'll replace this later

waveform = load_audio(audio_path)
waveform = preprocess_audio(waveform, target_length)

mel = extract_mel_spectrogram(waveform, target_sr)

patches = create_patches(mel)
patches = flatten_patches(patches)

print(waveform.shape)
print(mel.shape)
print(patches.shape)