import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
from spectral.io import envi
from scipy.ndimage import label
import glob
import time
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
    _ = np.where(np.asarray(Image.open(file_path))> 0, 1, 0).astype(np.uint8)   # this will make an 2D array with all positives=1
    return _*label

def label_transfer(file_path: str) -> list:
    '''used for transfer premixed colored groundtruth into class label represented in uint8'''
    _ = Image.open(file_path)
    _ = _.convert('RGB')
    labeled = []
    arr = np.asarray(_)                                                     # shape = (y,x,3)
    labeled.append(np.all(arr == (255,50,50), axis= -1).astype(np.uint8))   # class 1, shape = (y,x)
    labeled.append(np.all(arr == (50,255,50), axis= -1).astype(np.uint8)*2) # class 2

    return np.sum(labeled, axis=0, dtype= np.uint8)

def cut_hstack(size:tuple, data) -> tuple :
    '''
    This function is used for stacking multi images with only ONE pure class in each of them\n
    param:\n
    size = (y_len, x_len)\n
    data = list of (mask, hsi, (anchor_y, anchor_x))
    \n ret: ground truth stack, hsi stack
    '''
    y_len, x_len = size
    labeled = []
    hsis = []
    for index, (image, hsi, anchor) in enumerate(data):
        temp_label = image[anchor[0]:anchor[0]+y_len, anchor[1]:anchor[1]+x_len]
        labeled.append(np.all(temp_label != (0,0,0), axis= -1).astype(np.uint8)*(index+1))  # label transfer
        hsis.append(hsi[anchor[0]:anchor[0]+y_len, anchor[1]:anchor[1]+x_len, :])
    
    label_stack = np.hstack(labeled)
    hsi_stack = np.hstack(hsis)
    return label_stack, hsi_stack

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
    tic = time.time()
    print('start mapping as png')
    for ind, arr in enumerate(arrays):
        arr = np.where(arr==0, 17, arr)
        arr = arr[:,:]-1
        rav = np.ravel(arr)
        color = list_to_colormap(rav)
        color = np.reshape(color, (arr.shape[0], arr.shape[1], 3))
        classification_map(color, arr, 300, str(ind) + '.png')

    toc = time.time()
    print(f'finished mapping in {toc-tic} seconds')

def simple_select(cube: np.ndarray, denominator: int)-> np.ndarray:
    '''For decreasing band amounts in hsi'''
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

def labeled_filled_circle(shape:int, label:int, rad:int = None)->np.ndarray:
    '''returns a (shape*shape) 2D numpy array with 0 as background and label value as filled circle in center.\n
    radius of circle is set to 3/8 of shape if value of rad not given.'''
    y, x = np.ogrid[:shape, :shape]
    distance = np.sqrt((x - shape//2) ** 2 + (y - shape//2) ** 2)
    if rad is None:
        radius = shape*3//8
    else:
        radius = rad
    return np.where(distance <= radius, 1, 0).astype(np.uint8)*label

def blockize(hsi_with_mask:dict, patch_with_mask:dict, minor_label:int= 1, automask_rad = None)-> tuple[np.ndarray, np.ndarray]:
    '''hsi_with_mask: {'data':list of hsi, ''mask':list of ndarray mask}\n
    patch_with_mask: {'data':list of patch, ''mask':list of ndarray mask}\n
    @ hsi is not neccessary np.ndarray, an array like list should work
    @ None is acceptable for mask of patch
    @ return: hsi block, mask block'''

    print('start tidying up data')
    tic = time.time()
    # vstack main hsi block -------------------------------------------------------------
    major_block = np.vstack(hsi_with_mask['data'])  # width should be identical
    major_mask = np.vstack(hsi_with_mask['mask'])   # width should be identical

    PATCH_HEIGHT:int = patch_with_mask['data'][0].shape[0]
    BLOCK_HEIGHT:int = major_mask.shape[0]
    ROWMAX:int = BLOCK_HEIGHT//PATCH_HEIGHT
    # auto generate mask for patch mask where there are none
    patch_with_mask['mask'] = [labeled_filled_circle(PATCH_HEIGHT, minor_label, automask_rad) if patch is None
                               else patch for patch in patch_with_mask['mask']]

    # tile patches ----------------------------------------------------------------------
    remain_patch:int = len(patch_with_mask['mask'])
    minor_patch_tiles = []      # list to store tiles of patches
    minor_mask_tiles = []
    row_index:int = 0           # row counter

    while remain_patch > 0:
        if remain_patch >= ROWMAX:
            minor_patch_tiles.append(np.vstack(patch_with_mask['data'][ROWMAX*row_index:ROWMAX*(row_index+1)]))
            minor_mask_tiles.append(np.vstack(patch_with_mask['mask'][ROWMAX*row_index:ROWMAX*(row_index+1)]))
        else:
            incomplete_patch_tile = np.vstack(patch_with_mask['data'][ROWMAX*row_index:])
            incomplete_mask_tile = np.vstack(patch_with_mask['mask'][ROWMAX*row_index:])
            padtile3d = ((0,PATCH_HEIGHT*(ROWMAX-remain_patch)),(0,0),(0,0))
            padtile2d = ((0,PATCH_HEIGHT*(ROWMAX-remain_patch)),(0,0))
            minor_patch_tiles.append(np.pad(incomplete_patch_tile, padtile3d, mode='constant', constant_values= 0))
            minor_mask_tiles.append(np.pad(incomplete_mask_tile, padtile2d, mode='constant', constant_values= 0))
        # update counters
        row_index+= 1
        remain_patch-= ROWMAX

    minor_block = np.hstack(minor_patch_tiles)
    minor_mask = np.hstack(minor_mask_tiles)

    # make major and minor block same height
    minor_block = np.pad(minor_block, ((0,BLOCK_HEIGHT-PATCH_HEIGHT*ROWMAX),(0,0),(0,0)), mode='constant', constant_values= 0)
    minor_mask = np.pad(minor_mask, ((0,BLOCK_HEIGHT-PATCH_HEIGHT*ROWMAX),(0,0)), mode='constant', constant_values= 0)

    # hstack major and minor blocks
    final_block = np.hstack((major_block, minor_block))
    final_mask = np.hstack((major_mask, minor_mask))

    toc = time.time()
    print(f'finish tidying up data in {toc-tic} seconds')
    return final_block, final_mask

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    PATH = r'../data/'

    data = [((r'CTC\masks\20230617_v10-3.png', r'CTC\20230617_v10-3'),)
            , r'slices', r'slices\gen_img\3D\2000', r'slices\gen_img\3D\2100', r'slices\gen_img\3D\2200']

    def read(data):
        hsi_dict = {'data':[], 'mask':[]}
        patch_dict = {'data':[], 'mask':[]}
        for tup in data[0]:
            hsi_dict['mask'].append(label_transfer(PATH+tup[0]))
            envi_hsi = envi.open(PATH+tup[1] + ".hdr" , PATH+tup[1] + ".raw")
            hsi_dict['data'].append(envi_hsi.load())

        for patch_dir in data[1:]:
            npy_paths = glob.glob(PATH+patch_dir+r'/*.npy')
            for npy_path in npy_paths:
                patch_dict['data'].append(np.load(npy_path))
                patch_mask_path = npy_path.replace('.npy','_mask.png')
                if os.path.exists(patch_mask_path):
                    patch_dict['mask'].append(label_preprocess(patch_mask_path, 1))    # when class 1 is minor class
                else:
                    patch_dict['mask'].append(None)

        hsi, mask = blockize(hsi_dict, patch_dict)

        # -------------------------------------------
        direct_map(mask)

    read(data)