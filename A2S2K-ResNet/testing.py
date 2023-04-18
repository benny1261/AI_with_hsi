import os
import re
import torch
import torch.utils.data as Data
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import numpy as np
from sklearn import preprocessing

import Utils
import geniter
from A2S2KResNet import S3KAIResNet, load_dataset, PATCH_LENGTH

def extract_parameters(model_name: str):
    '''used to extract parameters from saved model file name'''

    key_bands = 'bands'
    key_classes = 'classes'
    bands = int(re.compile(f'(?<={key_bands})\d+').search(model_name).group())
    classes = int(re.compile(f'(?<={key_classes})\d+').search(model_name).group())
    return bands, classes

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # PATH = r'../data'
    data = (r'/mix/masks/1214_20x_wbc+A549_3.png', r'/mix/1214_20x_wbc+A549_3')
    MODEL = 'split0.9_lr0.001_adam_kernel24_bands30_classes2_0.999.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("inferencing on ", device)
    BATCH_SIZE = 32

    # Data Loading ------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_hsi, gt_hsi, TOTAL_SIZE = load_dataset(data)
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))           # flatten data
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )

    # Passive Params-----------------------------------------------------
    BANDS = data_hsi.shape[2]
    ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
    CLASSES_NUM = len(np.unique(gt))-1

    data = preprocessing.scale(data)            # standardize, equivalent to (X-X_mean)/X_std
    whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])          # shape back
    padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH),(0, 0)),'constant',constant_values=0)

    # load model --------------------------------------------------------
    model_band, model_class = extract_parameters(MODEL)
    if any((model_band != BANDS, model_class != CLASSES_NUM)):
        raise Exception("input data do not match model parameters")
    model = S3KAIResNet(BANDS, CLASSES_NUM, 2)
    state_dict = torch.load(r'models/'+ MODEL)              # if model saved on GPU need to add map_location= device
    model.load_state_dict(state_dict)
    model.to(device)                                        # move model to GPU which is saved on GPU previously
    # model.eval()                                          # seemed to be done in generate png function

    ''' 
    Make sure to call input = input.to(device) on any input tensors that you feed to the model
    Calling my_tensor.to(device) returns a new copy of my_tensor on GPU, it does NOT overwrite my_tensor
    Therefore, remember to manually overwrite tensors: my_tensor = my_tensor.to(torch.device('cuda'))
    '''
    _, total_indices = Utils.sampling(1, gt)
    gt_all = gt[total_indices] - 1
    all_data =  geniter.select_small_cubic(TOTAL_SIZE, total_indices, whole_data, PATCH_LENGTH, padded_data, BANDS)
    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], BANDS)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)

    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False,
        num_workers=0,
    )

    Utils.generate_png(all_iter, model, gt_hsi, device, total_indices, '')