import torch
import scipy.io as sio
import os
import numpy as np
from sklearn.decomposition import PCA

x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.chdir(os.path.dirname(__file__))
data_path = r'data/'
mat_data = sio.loadmat(data_path + 'Indian_pines_corrected.mat')
mat_gt = sio.loadmat(data_path + 'Indian_pines_gt.mat')
data_hsi = mat_data['indian_pines_corrected']
gt_hsi = mat_gt['indian_pines_gt'] 

K = 10
shapeorig = data_hsi.shape
data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
data_hsi = PCA(n_components= K).fit_transform(data_hsi)
shapeorig = np.array(shapeorig)
shapeorig[-1] = K
data_hsi = data_hsi.reshape(shapeorig)

print(gt_hsi.shape)
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
print(gt.shape)
print(max(gt))