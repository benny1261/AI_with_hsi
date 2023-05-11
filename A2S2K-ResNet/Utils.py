import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import spectral
from spectral.io import envi
from scipy.ndimage import label
# from sklearn import metrics, preprocessing
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = len(np.unique(ground_truth))-1
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
            y[index] = np.array([255, 50, 50]) / 255.   # changed
        if item == 1:
            y[index] = np.array([50, 255, 50]) / 255.   # changed
        if item == 2:
            y[index] = np.array([50, 50, 255]) / 255.   # changed
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
    '''image to numpy array with pixel value replaces by intergers representing its class, dtype=uint8
    \n used for non-premixed samples'''
    _ = np.asarray(Image.open(file_path))   # this will make an 2D array with all nonzeros=1
    return _*label

def label_transfer(file_path: str) -> list:
    '''used for transfer premixed colored groundtruth into class label represented in uint8'''
    _ = Image.open(file_path)
    _ = _.convert('RGB')
    # _.show()
    labeled = []
    arr = np.asarray(_)                                                     # shape = (y,x,3)
    labeled.append(np.all(arr == (255,50,50), axis= -1).astype(np.uint8))   # class 1, shape = (y,x)
    labeled.append(np.all(arr == (50,255,50), axis= -1).astype(np.uint8)*2) # class 2

    return np.sum(labeled, axis=0, dtype= np.uint8)

def cut_hstack(size:tuple, data) -> tuple :
    '''param: data = list of (grayscaleimage, hsi, (anchor_y, anchor_x))
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

def cut_merge(size:tuple, data) -> tuple :
    '''param: data = list of (grayscaleimage, hsi, (anchor_y, anchor_x))
    \n ret: ground truth merge, hsi merge
    '''
    HSI_ORIG_BANDS = 150
    y_len, x_len = size
    struc_kernal = np.ones((3, 3), dtype=np.uint0)
    labeled = []
    hsis = []
    for image, hsi, anchor in data:
        labeled.append(image[anchor[0]:anchor[0]+y_len, anchor[1]:anchor[1]+x_len])
        hsis.append(hsi[anchor[0]:anchor[0]+y_len, anchor[1]:anchor[1]+x_len, :])

    # classnum = len(labeled) ///currently only 2 classes available///
    hsi_mix = hsis[0]                                                       # uses first hsi image as background
    onlyc2_mask = np.logical_and(np.logical_not(labeled[0]), labeled[1])    # 2d mask
    cyx_cube = np.moveaxis(np.empty(hsis[0].shape, dtype=bool), 2, 0)       # shape = (channel, spatial_y, spatial_x)
    cyx_cube[:,:,:] = onlyc2_mask
    yxc_cube = np.moveaxis(cyx_cube, 0, -1)
    np.putmask(hsi_mix, yxc_cube, hsis[1])

    overlap = np.logical_and.reduce(labeled).astype(np.uint8)  # = np.logical_and(np.logical_and(x, y), z)
    island_labeled, num_islands = label(overlap, structure=struc_kernal)
    mixed = np.sum(labeled, axis=0, dtype= np.uint8)
    for _ in range(num_islands):
        island_index = _+1
        class_rand = random.randint(1,2)
        island_mask = island_labeled == island_index
        np.putmask(mixed, island_mask, class_rand)

        cyx_cube = np.moveaxis(np.empty(hsis[0].shape, dtype=bool), 2, 0)
        cyx_cube[:,:,:] = island_mask
        yxc_cube = np.moveaxis(cyx_cube, 0, -1)
        np.putmask(hsi_mix, yxc_cube, hsis[class_rand-1])

    # direct_map(mixed, labeled[0], labeled[1])
    return mixed, hsi_mix

def direct_map(*arrays):
    '''used for quick visualizing label or predictions'''
    for ind, arr in enumerate(arrays):
        arr = np.where(arr==0, 17, arr)
        arr = arr[:,:]-1
        rav = np.ravel(arr)
        color = list_to_colormap(rav)
        color = np.reshape(color, (arr.shape[0], arr.shape[1], 3))
        classification_map(color, arr, 300, str(ind) + '.png')

def simple_select(cube: np.ndarray, denominator: int)-> np.ndarray:
    limit = cube.shape[-1]
    def recursive_stack(stack, band:int= 1)-> list:
        '''initial stack should be band 0'''
        if band == limit:
            return stack
        elif band % denominator == 0:
            stack = np.concatenate((stack, cube[:,:,band]), axis= 2)
            return recursive_stack(stack, band+1)
        else:
            return recursive_stack(stack, band+1)

    # block invalid value
    if any((not isinstance(denominator, int), denominator< 1, denominator> limit)):
        raise ValueError
    elif denominator == 1:
        return np.array(cube)

    return np.array(recursive_stack(cube[:,:,0], 1))


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    PATH = r'../data/'
    CUT_SIZE = (1000, 2000)
    spectral.settings.envi_support_nonlowercase_params = 'TRUE'

    data = (r'/mix/masks/1214_20x_wbc+A549_3.png', r'/mix/1214_20x_wbc+A549_3')

    # for _ in range(len(data)):
    #     label_path, hsi_path, anchor = data[_]
    #     labeled = label_preprocess(PATH+label_path, _+1)  # auto labeling
    #     hsi = envi.open(PATH+hsi_path + ".hdr" , PATH+hsi_path + ".raw")
    #     data[_] = (labeled, hsi, anchor)

    # _ = np.asarray(Image.open(PATH+data[0]))
    # print(_.shape)
    gt_hsi = label_transfer(PATH+data[0])
    data_hsi = envi.open(PATH+data[1] + ".hdr" , PATH+data[1] + ".raw")