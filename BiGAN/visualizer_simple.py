# -*- coding: utf-8 -*-
'''
Simple Visualiser
'''

from torch.utils import data
from torchvision import datasets, transforms, utils
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

mold_img = ImageFolder('D:/Desk/Kandidatspeciale - Ganspection/Kode/GANspection/BiGAN/data/folder/train', transform)

loader = data.DataLoader(mold_img, BATCH_SIZE, shuffle=True)

def imshow(img):
    """ show an image """
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


train_data_iter = iter(loader)

inputs, classes = next(iter(loader)) 

# get some random training images
images, labels = train_data_iter.next()

# show images
imshow(torchvision.utils.make_grid(images))
