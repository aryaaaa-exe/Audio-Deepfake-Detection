import torch
import numpy as np

#Creating Patches
def create_patches(mel_spectrogram):
  patch_size=16
  patches=[] #empty list

  for i in range(0,mel_spectrogram.shape[0],patch_size): #0,16,32,48,64; starting rows
    for j in range(0,mel_spectrogram.shape[1],patch_size): #0,16,32,...,96; (32 positions in total)
      #Extract path
      patch=mel_spectrogram[
          i:i+patch_size,
          j:j+patch_size
      ]
      #storing the patches in a list
      patches.append(patch)

  #list converted to numpy array to give a shape of (patch number,size,size)
  #neural networks expect array/tensors so conversion is useful
  patches=np.array(patches)

  #Converts Numpy to Tensor
  patches=torch.tensor(
      patches,
      dtype=torch.float32 #Neural networks perform calculations using floating-point numbers so it is needed to specify the type
  )

  return patches

#Flattening the patches
def flatten_patches(patches):
  patches= patches.reshape(patches.shape[0], -1)
  #print("After Flattening: ", patches.shape)
  return patches
