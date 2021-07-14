# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 20:29:26 2021

@author: lazar
"""

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from bigan_utils import DeterministicConditional, JointCritic, WALI
from dataload import MVTecDataset

# =============================================================================
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True
# torch.manual_seed(1)
# torch.cuda.manual_seed_all(1)
# =============================================================================

# training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 1000
IMAGE_SIZE = 32
NUM_CHANNELS = 3
DIM = 128
NLAT = 100
LEAK = 0.2

C_ITERS = 5       # critic iterations
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.9

FEATURES_CRITIC = 16
FEATURES_GEN = 16

def create_encoder():
  mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    Conv2d(DIM * 4, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    Conv2d(DIM * 4, NLAT, 1, 1, 0))
  return DeterministicConditional(mapping)


def create_generator():
  mapping = nn.Sequential(
    ConvTranspose2d(NLAT, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
  return DeterministicConditional(mapping)


def create_critic():
  x_mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, DIM, 4, 2, 1), LeakyReLU(LEAK),
    Conv2d(DIM, DIM * 2, 4, 2, 1), LeakyReLU(LEAK),
    Conv2d(DIM * 2, DIM * 4, 4, 2, 1), LeakyReLU(LEAK),
    Conv2d(DIM * 4, DIM * 4, 4, 1, 0), LeakyReLU(LEAK))

  z_mapping = nn.Sequential(
    Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))

  joint_mapping = nn.Sequential(
    Conv2d(DIM * 4 + 512, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 1024, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(1024, 1, 1, 1, 0))

  return JointCritic(x_mapping, z_mapping, joint_mapping)


def create_WALI():
  E = create_encoder()
  G = create_generator()
  C = create_critic()
  wali = WALI(E, G, C)
  return wali


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
wali = create_WALI().to(device)

optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()), 
  lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizerC = Adam(wali.get_critic_parameters(), 
  lr=LEARNING_RATE, betas=(BETA1, BETA2))

dataset = MVTecDataset(class_name='hazelnut', is_train=True, resize=IMAGE_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

fixed_noise = torch.randn(16, NLAT, 1, 1).to(device)
writer = SummaryWriter()

curr_iter = C_iter = EG_iter = 0
C_update, EG_update = True, False
print('Training starts...')

for epoch in range(NUM_EPOCHS):
    for batch_idx, (x, _, _) in enumerate(loader):
      x = x.to(device)
      
      if epoch == 0 and batch_idx == 0:
        init_x = x
      
      z = torch.randn(x.size(0), NLAT, 1, 1).to(device)
      C_loss, EG_loss = wali(x, z, lamb=LAMBDA)
      
      if C_update:
        optimizerC.zero_grad()
        C_loss.backward()
        optimizerC.step()
        C_iter += 1
      
        if C_iter == C_ITERS:
          C_iter = 0
          C_update, EG_update = False, True
        continue
      
      if EG_update:
        optimizerEG.zero_grad()
        EG_loss.backward()
        optimizerEG.step()
        EG_iter += 1
      
        if EG_iter == EG_ITERS:
          EG_iter = 0
          C_update, EG_update = True, False
          curr_iter += 1
        else:
          continue

    # print training statistics
    if epoch % 100 == 0:
        wali.eval()
        print( f"Epoch {epoch}" )
        writer.add_scalar("Loss/Train_Generator/Encoder",EG_loss.item() , epoch*len(dataset) + batch_idx*BATCH_SIZE)
        writer.add_scalar("Loss/Train_Critic", C_loss.item(), epoch*len(dataset) + batch_idx*BATCH_SIZE)
        with torch.no_grad():
            fake = wali.generate(fixed_noise)
            rect = wali.reconstruct(init_x[:32])
            # take out (up to) 32 examples
            img_grid_real = torchvision.utils.make_grid(init_x[:32], normalize=True)
            img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
            img_grid_rect = torchvision.utils.make_grid(rect[:32], normalize=True)
            writer.add_image("Real", img_grid_real, global_step=epoch)
            writer.add_image("Fake", img_grid_fake, global_step=epoch)
            writer.add_image("Rect", img_grid_rect, global_step=epoch)
        wali.train()
