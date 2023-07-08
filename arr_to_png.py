import os
import glob
import numpy as np
import torch
from torchvision.utils import save_image

PATH = os.path.join(os.path.dirname(__file__), r'data/slices')

os.makedirs(os.path.join(PATH,"flatten"), exist_ok=True)
file_paths = glob.glob(os.path.join(PATH, '*.npy'))
print(f'{len(file_paths)} numpy files found')
for file_path in file_paths:
    tensor = torch.from_numpy(np.load(file_path))       # HxWxC
    tensor = tensor.permute(2, 0, 1).unsqueeze(1)       # Cx1xHxW
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_image(tensor.data, os.path.join(PATH,"flatten",f"{file_name}.png"), nrow=15, normalize=True)