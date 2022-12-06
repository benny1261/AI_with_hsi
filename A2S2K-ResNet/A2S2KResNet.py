import collections
import math
import time
import os

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics, preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import torch_optimizer as optim2
from torchsummary import summary
import spectral
from spectral.io import envi

import geniter
import record
import Utils

# Setting Parameters ------------------------------------------------
VERIFY: bool = False
PATH = r'../data/'
data = [(r'hct8/masks/1018_2_hct8.png', r'hct8/1018_2_processed_fixed', (0,0)),
        (r'nih3t3/masks/1018_2_nih3t3.png', r'nih3t3/1018_2_processed_fixed', (0,0))]
CUT_SIZE = (300, 200)
REMAIN_BAND: int = 30            # number of channels to keep
VALIDATION_SPLIT = 0.9
ITER: int = 1
PATCH_LENGTH: int = 4
KERNEL_SIZE: int = 24
lr, num_epochs, batch_size = 0.001, 200, 32
loss = torch.nn.CrossEntropyLoss()
OPTIM = 'adam'
EPOCH: int = 40
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]      # for Monte Carlo runs

# Data Loading ------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("training on ", device)

def load_dataset(data): # originally parameters are used for decide which data to load

    if VERIFY:
        mat_data = sio.loadmat(PATH + 'Indian_pines_corrected.mat')
        mat_gt = sio.loadmat(PATH + 'Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']           # 145*145*200
        gt_hsi = mat_gt['indian_pines_gt']                      # 145*145
    else:
        spectral.settings.envi_support_nonlowercase_params = 'TRUE'
        for _ in range(len(data)):
            label_path, hsi_path, anchor = data[_]
            labeled = Utils.label_preprocess(PATH+label_path, _+1)  # auto labeling
            hsi = envi.open(PATH+hsi_path + ".hdr" , PATH+hsi_path + ".raw")
            data[_] = (labeled, hsi, anchor)

        gt_hsi, data_hsi = Utils.cut_combine(CUT_SIZE, *data)
        print(gt_hsi.shape, data_hsi.shape)

    shapeorig = data_hsi.shape
    data_hsi = data_hsi.reshape(-1, data_hsi.shape[-1])
    data_hsi = PCA(n_components=REMAIN_BAND).fit_transform(data_hsi)
    shapeorig = np.array(shapeorig)                         # turn tuple into list -> value assignable
    shapeorig[-1] = REMAIN_BAND
    data_hsi = data_hsi.reshape(shapeorig)

    nonzero_number_size = np.count_nonzero(gt_hsi)      # 10249 in example
    return data_hsi, gt_hsi, nonzero_number_size

os.chdir(os.path.dirname(__file__))
data_hsi, gt_hsi, TOTAL_SIZE = load_dataset(data)
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )

# Passive Params-----------------------------------------------------
img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1
BANDS = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]
CLASSES_NUM = len(np.unique(gt))-1
print('The class numbers of the HSI data is:', CLASSES_NUM)

# Container Params---------------------------------------------------
KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)            # standardize, equivalent to (X-X_mean)/X_std
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])  # shape back
# pad before convolution to prevent decrease spatial dimension length
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH),(0, 0)),'constant',constant_values=0)

# Model -------------------------------------------------------------
class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))
        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))
        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))
        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([
            squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
            squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
            squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)
        ])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)
        return output_tensor


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(-1, -3).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding,
            use_1x1conv=False, stride= 1, start_block=False, end_block=False,):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU())
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        # ECA Attention Layer
        self.ecalayer = eca_layer(out_channels)

        # start and end block initialization
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        out = self.ecalayer(out)

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out


class S3KAIResNet(nn.Module):
    def __init__(self, band, classes, reduction):
        super(S3KAIResNet, self).__init__()
        self.conv1x1 = nn.Conv3d(in_channels=1, out_channels= KERNEL_SIZE, kernel_size=(1, 1, 7), stride=(1, 1, 2), padding=0)
        self.conv3x3 = nn.Conv3d(in_channels=1, out_channels= KERNEL_SIZE, kernel_size=(3, 3, 7), stride=(1, 1, 2), padding=(1, 1, 0))

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(KERNEL_SIZE, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(band // reduction, KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = Residual(KERNEL_SIZE, KERNEL_SIZE, (1, 1, 7), (0, 0, 3), start_block=True)
        self.res_net2 = Residual(KERNEL_SIZE, KERNEL_SIZE,(1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(KERNEL_SIZE, KERNEL_SIZE,(3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(KERNEL_SIZE, KERNEL_SIZE, (3, 3, 1), (1, 1, 0), end_block=True)
        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)

        self.conv2 = nn.Conv3d(in_channels= KERNEL_SIZE, out_channels=128, padding=(0, 0, 0), kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(in_channels=1, out_channels= KERNEL_SIZE, padding=(0, 0, 0), kernel_size=(3, 3, 128), stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(KERNEL_SIZE, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True))

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(       # single layer linear classifier
            nn.Linear(KERNEL_SIZE, classes)
            # nn.Softmax()
        )

    def forward(self, X):
        x_1x1 = self.conv1x1(X)
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3x3(X)
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)

        x1 = torch.cat([x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat(
            [
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1)
            ],
            dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)

        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        return self.full_connection(x4)

def train(net, train_iter, valida_iter, loss, optimizer, device, epochs, early_stopping=True, early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)

        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step()
        valida_acc, valida_loss = record.evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)

        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print(
            'epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
            % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,
               valida_loss, valida_acc, time.time() - time_epoch))

        MODEL_PATH = "./net_DBA.pt"

        if early_stopping and loss_list[-2] < loss_list[-1]:
            if early_epoch == 0:
                torch.save(net.state_dict(), MODEL_PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(MODEL_PATH))
                break
        else:
            early_epoch = 0

    print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
        % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))


model = S3KAIResNet(BANDS, CLASSES_NUM, 2).cuda()
summary(model, input_size=(1, img_rows, img_cols, BANDS))

# Training ----------------------------------------------------------
for index_iter in range(ITER):
    #define the model
    net = S3KAIResNet(BANDS, CLASSES_NUM, 2)

    if OPTIM == 'diffgrad':
        optimizer = optim2.DiffGrad(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)  # weight_decay=0.0001)
    if OPTIM == 'adam':
        optimizer = optim.Adam(
            net.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])

    train_indices, test_indices = Utils.sampling(VALIDATION_SPLIT, gt)
    _, total_indices = Utils.sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
    print('Test size: ', TEST_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    # Pytorch DataLoader
    train_iter, valida_iter, test_iter, all_iter = geniter.generate_iter(
        TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE,
        total_indices, VAL_SIZE, whole_data, PATCH_LENGTH, padded_data, BANDS, batch_size, gt)

    print(f'------training({index_iter+1}/{ITER})------')
    tic1 = time.time()
    train(net, train_iter, valida_iter, loss, optimizer, device, epochs= EPOCH)
    toc1 = time.time()

    pred_test = []
    tic2 = time.time()
    with torch.no_grad():
        for X, y in test_iter:
            # print('Shape of X', X.shape, 'Shape of y', y.shape)
            # X = X.permute(0, 3, 1, 2)
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test.extend(np.array(y_hat.cpu().argmax(axis=1)))
    toc2 = time.time()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1

    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = record.aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(
        net.state_dict(), "./models/" + 'split_' + str(VALIDATION_SPLIT) + '_lr_' + str(lr) + '_'
        + OPTIM + '_kernel_' + str(KERNEL_SIZE) + '_' + str(round(overall_acc, 3)) + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc

# Map & Records
print("--------" + " Training Finished-----------")
if not os.path.exists('report'):
    os.makedirs('report')
record.record_output(
    OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
    './report/' + str(img_rows) + '_' + 'split' + str(VALIDATION_SPLIT) + '_lr' +
    str(lr) + '_' + OPTIM + '_kernel' + str(KERNEL_SIZE) + '.txt')

if not os.path.exists('classification_maps'):
    os.makedirs('classification_maps')
Utils.generate_png(
    all_iter, net, gt_hsi, device, total_indices,
    './classification_maps/' + str(img_rows) + '_' + 'split' + str(VALIDATION_SPLIT) +
    '_lr' + str(lr) + '_' + OPTIM + '_kernel' + str(KERNEL_SIZE))