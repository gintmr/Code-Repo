import os 
import torch


pth_path = '/data2/wuxinrui/OpenGDINO_DisC/outputs/checkpoint.pth'

weights = torch.load(pth_path)

for k, v in weights.items():
    print(k)