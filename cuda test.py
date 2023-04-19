import torch
import scipy.io as sio
import os
import numpy as np
from sklearn.decomposition import PCA
import shutil

# x = torch.rand(5, 3)
# print(x)
# print(torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir(os.path.dirname(__file__))
data_path = r'data/'