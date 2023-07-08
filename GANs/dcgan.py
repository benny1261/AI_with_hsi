import os
import glob
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
# from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

EPOCH:int = 2000
BATCH_SIZE:int = 16             # orig 64
LR:float = 0.0002
B1:float = 0.5                  # adam: decay of first order momentum of gradient
B2:float = 0.999                # adam: decay of first order momentum of gradient
LATENT_DIM:int = 100
IMG_SIZE:int = 32               # size of image spatial dimension
CHANNELS:int = 150              # number of image channels
SAMPLE_INTERVAL:int = 100       # interval between image sampling

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)     # Gaussaian distribution noise vector (latent vectors)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)     # Gaussaian distribution noise vector (latent vectors)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = IMG_SIZE // 4
        self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.kernel_size:int = 3
        self.stride:int = 2
        self.padding:int = 1

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, self.kernel_size, self.stride, self.padding),
                     nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(CHANNELS, 16, bn=False),   # 8x16x20x20
            *discriminator_block(16, 32),                   # 8x32x10x10
            *discriminator_block(32, 64),                   # 8x64x5x5
            *discriminator_block(64, 128),                  # 8x128x3x3
        )

        def calculate_downsample(input_size:int, kernel= self.kernel_size, stride= self.stride, padding= self.padding, recursive= 1)-> int:
            if recursive == 1:
                return (input_size - kernel + 2 * padding) // stride + 1                
            else:
                new_size = calculate_downsample(input_size)
                return calculate_downsample(new_size, kernel= kernel, stride= stride, padding= padding, recursive= recursive-1)

        # The height and width of downsampled image
        ds_size = calculate_downsample(IMG_SIZE, recursive= 4)
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class CustomDataset(Dataset):
    '''custom dataset to load only data(without labeled data)\n
    input format is: N x H x W x C'''
    def __init__(self, data, transform= None):
        super().__init__()
        self.data = np.transpose(np.array(data), (0, 3, 1, 2))  # permute to pytorch default C x H x W format

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

# input
PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), r'data/slices')
os.makedirs(os.path.join(PATH,"gen_img"), exist_ok=True)
file_paths = glob.glob(os.path.join(PATH, '*.npy'))
npy_list = []
for file_path in file_paths:
    npy_list.append(np.load(file_path))
print(f'{len(npy_list)} file loaded')

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

cuda = True if torch.cuda.is_available() else False

if cuda:
    print('cuda available')
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
mydataset = CustomDataset(npy_list,transform= transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=[0.5]*CHANNELS, std=[0.5]*CHANNELS)]
                         ))
dataloader = DataLoader(
    mydataset,
    batch_size= BATCH_SIZE,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(B1, B2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(B1, B2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(EPOCH):
    for i, imgs in enumerate(dataloader):       # I made custom dataloader return only data (without label)

        # Adversarial ground truths
        valid = Tensor(imgs.shape[0], 1).fill_(1.0)
        fake = Tensor(imgs.shape[0], 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM)))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, EPOCH, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % SAMPLE_INTERVAL == 0:
            # per_imgs = gen_imgs.permute(0, 2, 3, 1)
            save_image(gen_imgs.data[:,49:50,:,:], os.path.join(PATH,"gen_img","%d.png") % batches_done, nrow=4, normalize=True)