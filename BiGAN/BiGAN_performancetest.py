# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 18:11:10 2020

@author: lazar
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from BiGAN_test import create_encoder, create_generator, create_critic, IMAGE_SIZE, NLAT
# import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from utils import DeterministicConditional, GaussianConditional, JointCritic, WALI
from torchvision import datasets, transforms, utils
from torchvision.datasets import ImageFolder
plt.close('all')
'''
struct:
    
    load test data
    load model
    
    loop:
        generate data and labels
        
    train seperator model OR anomaly score: L1- recon og org + distance in lat-space
    
'''

def create_WALI():
  E = create_encoder()
  G = create_generator()
  C = create_critic()
  wali = WALI(E, G, C)
  return wali

# Testdata load
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])
    
mold_img_test = ImageFolder('D:/Desk/Kandidatspeciale - Ganspection/Kode/GANspection/BiGAN/data/folder/test', transform)
loader = data.DataLoader(mold_img_test)

# Model load
final_iter = 900
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
wali = create_WALI().to(device)

wali.load_state_dict(torch.load('Runs/%d.ckpt' % final_iter))
wali.eval()

# Generate data for classifier
inputs, labels = torch.zeros(1,IMAGE_SIZE,IMAGE_SIZE).to(device), torch.zeros(1, dtype=torch.long).to(device)
lat, rec = torch.zeros(1,NLAT).to(device),torch.zeros(1,IMAGE_SIZE,IMAGE_SIZE).to(device)
lat_rec = torch.zeros(1,NLAT).to(device)
#lat_dist, rec_dist = torch.zeros(1).to(device), torch.zeros(1).to(device)
lat_dist, rec_dist = [], []
print('Testing starts...')
for batch_idx, (x, label) in enumerate(loader, 1):
    
    label = label.to(device)
    x = x.to(device)
    labels = torch.cat((labels, label))
    inputs = torch.cat((inputs, x.squeeze(1)), 0)
    
    rec = torch.cat((rec, wali.reconstruct(x).squeeze(1)), 0)
    lat = torch.cat((lat, wali.encode(x).squeeze(2).squeeze(2)), 0)
    lat_rec = torch.cat((lat_rec, wali.encode(wali.reconstruct(x)).squeeze(2).squeeze(2)), 0)
    
    rec_dist_temp = torch.sum(torch.abs(x.squeeze()-wali.reconstruct(x).squeeze())).item()
    rec_dist.append(rec_dist_temp)
    
    lat_dist_temp = torch.sum(torch.abs( (wali.encode(x).squeeze(2).squeeze(2)-wali.encode(wali.reconstruct(x)).squeeze(2).squeeze(2)).detach() )).item()
    lat_dist.append(lat_dist_temp)

# removing forst 0-layers from initialization. Stupid approch.
labels = labels[1:]
inputs = inputs[1:,:,:]
rec = rec[1:,:,:]
lat = lat[1:,:]
lat_rec = lat_rec[1:,:]

lat_dist_ar = np.array(lat_dist)
plt.figure()
plt.plot(lat_dist_ar,'.')
plt.title('latent space distance')

rec_dist_ar = np.array(rec_dist)
plt.figure()
plt.plot(rec_dist_ar,'.')
plt.title('reconstruction distance')

n_normal = len(labels)-sum(labels).item()
n_abnormal = len(labels)-n_normal

A_score = lat_dist_ar/max(lat_dist_ar)+rec_dist_ar/max(rec_dist_ar)
plt.figure()
plt.plot(A_score[:n_normal],'.', label='normal')
plt.plot(A_score[n_normal:],'.', label='abnormal')
plt.title('Anomaly score')
plt.legend()

plt.figure()
plt.hist(A_score[:n_normal], label='normal', alpha = 0.5)
plt.hist(A_score[n_normal:], label='abnormal', alpha = 0.5)
plt.legend()
plt.title('histogram of anomaly score')