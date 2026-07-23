import torch
from model.ps3dt import PS3DT

# Create the model
model = PS3DT()   # We may need to change this depending on your constructor

# Load weights
state_dict = torch.load("weights/best_param.pth", map_location="cpu")

model.load_state_dict(state_dict)

model.eval()

print("Model loaded successfully!")