import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import spectral
from spectral.io import envi
# from sklearn import metrics, preprocessing
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
# from operator import truediv
# import scipy.io as sio
# import torch


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist())if x == i + 1] # will not take background:0 into account
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)   # preserves at least 3 training data
        else:
            nb_val = 0
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        # print(len(train[i]), len(test[i]))
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes


# def set_figsize(figsize=(3.5, 2.5)):
#     display.set_matplotlib_formats('svg')
#     plt.rcParams['figure.figsize'] = figsize


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] / dpi, ground_truth.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)
    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_png(all_iter, net, gt_hsi, device, total_indices, path):
    pred_test = []
    for X, y in all_iter:
        #X = X.permute(0, 3, 1, 2)
        X = X.to(device)
        net.eval()
        pred_test.extend(net(X).cpu().argmax(axis=1).detach().numpy())
    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:          # if gt background
            gt[i] = 17          # 17-1=16 -> black
            x_label[i] = 16     # 16 -> black
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)
    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)
    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    classification_map(y_re, gt_hsi, 300, path + '.png')
    classification_map(gt_re, gt_hsi, 300, path + '_gt.png')
    print('------Get classification maps successful-------')

def label_preprocess(file_path: str, label: int) -> list:
    '''image to numpy array with pixel value replaces by intergers representing its class, dtype=uint8'''
    _ = np.asarray(Image.open(file_path))   # this will make an 2D array with all nonzeros=1
    return _*label

def cut_combine(size:tuple, *data) -> tuple :
    '''param: *data = (grayscaleimage, hsi, (anchor_y, anchor_x))
    \n ret: ground truth stack, hsi stack
    '''
    y_len, x_len = size
    labeled = []
    hsis = []
    for image, hsi, anchor in data:
        labeled.append(image[anchor[0]:anchor[0]+y_len, anchor[1]:anchor[1]+x_len])
        hsis.append(hsi[anchor[0]:anchor[0]+y_len, anchor[1]:anchor[1]+x_len, :])
    
    lab_stack = np.hstack(labeled)
    hsi_stack = np.hstack(hsis)
    return lab_stack, hsi_stack

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    PATH = r'../data/'
    spectral.settings.envi_support_nonlowercase_params = 'TRUE'

    labeled1 = label_preprocess(PATH+r'hct8/masks/1018_2_hct8.png', 1)
    labeled2 = label_preprocess(PATH+r'nih3t3/masks/1018_2_nih3t3.png', 2)
    hsi_1 = envi.open(PATH+r'hct8/1018_2_processed_fixed' + ".hdr" , PATH+r'hct8/1018_2_processed_fixed' + ".raw")
    hsi_2 = envi.open(PATH+r'nih3t3/1018_2_processed_fixed' + ".hdr" , PATH+r'nih3t3/1018_2_processed_fixed' + ".raw")

    print(labeled2.shape)
    print(hsi_1.shape)
    lb, hsi = cut_combine((300,200), (labeled1, hsi_1, (0,0)), (labeled2, hsi_2, (0,0)))
    print(lb.shape, hsi.shape)