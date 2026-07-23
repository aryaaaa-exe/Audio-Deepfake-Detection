import torch

checkpoint = torch.load("weights/best_param.pth", map_location="cpu")

print(type(checkpoint))

if isinstance(checkpoint, dict):
    print(checkpoint.keys())