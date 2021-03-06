# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:29:20 2020

@author: lazar
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils import data
from torch.nn import Conv2d, ConvTranspose2d, BatchNorm2d, LeakyReLU, ReLU, Tanh
from utils import DeterministicConditional, GaussianConditional, JointCritic, WALI
from torchvision import datasets, transforms, utils
from torchvision.datasets import ImageFolder

cudnn.benchmark = True
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

# training hyperparameters
BATCH_SIZE = 64
ITER = 200
IMAGE_SIZE = 32
NUM_CHANNELS = 1
DIM = 128
NLAT = 100
LEAK = 0.2

C_ITERS = 5       # critic iterations
EG_ITERS = 1      # encoder / generator iterations
LAMBDA = 10       # strength of gradient penalty
LEARNING_RATE = 1e-4
BETA1 = 0.5
BETA2 = 0.9

# def create_encoder():
#   mapping = nn.Sequential(
#     Conv2d(NUM_CHANNELS, 32, 5, 1, bias=False), BatchNorm2d(32), LeakyReLU(LEAK, inplace=True),
#     Conv2d(32, 64, 4, 2, bias=False), BatchNorm2d(64), LeakyReLU(LEAK, inplace=True),
#     Conv2d(64, 128, 4, 1, bias=False), BatchNorm2d(128), LeakyReLU(LEAK, inplace=True),
#     Conv2d(128, 256, 4, 2, bias=False), BatchNorm2d(256), LeakyReLU(LEAK, inplace=True),
#     Conv2d(256, 512, 4, 1, bias=False), BatchNorm2d(512), LeakyReLU(LEAK, inplace=True),
#     Conv2d(512, 512, 1, 1, bias=False), BatchNorm2d(512), LeakyReLU(LEAK, inplace=True),
#     Conv2d(512, 2 * NLAT, 1, 1))
#   return GaussianConditional(mapping)

def create_encoder():
  mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    Conv2d(DIM, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    Conv2d(DIM * 2, DIM * 4, 4, 2, 1, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    Conv2d(DIM * 4, NLAT, 4, 1, 0, bias=False))
  return DeterministicConditional(mapping)

# def create_generator():
#   mapping = nn.Sequential(
#     ConvTranspose2d(NLAT, 256, 4, 1, bias=False), BatchNorm2d(256), LeakyReLU(LEAK, inplace=True),
#     ConvTranspose2d(256, 128, 4, 2, bias=False), BatchNorm2d(128), LeakyReLU(LEAK, inplace=True),
#     ConvTranspose2d(128, 64, 4, 1, bias=False), BatchNorm2d(64), LeakyReLU(LEAK, inplace=True),
#     ConvTranspose2d(64, 32, 4, 2, bias=False), BatchNorm2d(32), LeakyReLU(LEAK, inplace=True),
#     ConvTranspose2d(32, 32, 5, 1, bias=False), BatchNorm2d(32), LeakyReLU(LEAK, inplace=True),
#     ConvTranspose2d(32, 32, 1, 1, bias=False), BatchNorm2d(32), LeakyReLU(LEAK, inplace=True),
#     Conv2d(32, NUM_CHANNELS, 1, 1, bias=False), Tanh())
#   return DeterministicConditional(mapping)

def create_generator():
  mapping = nn.Sequential(
    ConvTranspose2d(NLAT, DIM * 4, 4, 1, 0, bias=False), BatchNorm2d(DIM * 4), ReLU(inplace=True),
    ConvTranspose2d(DIM * 4, DIM * 2, 4, 2, 1, bias=False), BatchNorm2d(DIM * 2), ReLU(inplace=True),
    ConvTranspose2d(DIM * 2, DIM, 4, 2, 1, bias=False), BatchNorm2d(DIM), ReLU(inplace=True),
    ConvTranspose2d(DIM, NUM_CHANNELS, 4, 2, 1, bias=False), Tanh())
  return DeterministicConditional(mapping)

# def create_critic():
#   x_mapping = nn.Sequential(
#     Conv2d(NUM_CHANNELS, 32, 5, 1), LeakyReLU(LEAK),
#     Conv2d(32, 64, 4, 2), LeakyReLU(LEAK),
#     Conv2d(64, 128, 4, 1), LeakyReLU(LEAK),
#     Conv2d(128, 256, 4, 2), LeakyReLU(LEAK),
#     Conv2d(256, 512, 4, 1), LeakyReLU(LEAK))

#   z_mapping = nn.Sequential(
#     Conv2d(NLAT, 512, 1, 1, bias=False), LeakyReLU(LEAK),
#     Conv2d(512, 512, 1, 1, bias=False), LeakyReLU(LEAK))

#   joint_mapping = nn.Sequential(
#     Conv2d(1024, 1024, 1, 1), LeakyReLU(LEAK),
#     Conv2d(1024, 1024, 1, 1), LeakyReLU(LEAK),
#     Conv2d(1024, 1, 1, 1))

#   return JointCritic(x_mapping, z_mapping, joint_mapping)

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


def main():
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  wali = create_WALI().to(device)

  optimizerEG = Adam(list(wali.get_encoder_parameters()) + list(wali.get_generator_parameters()),
    lr=LEARNING_RATE, betas=(BETA1, BETA2))
  optimizerC = Adam(wali.get_critic_parameters(),
    lr=LEARNING_RATE, betas=(BETA1, BETA2))

# for 3 channels
# =============================================================================
#   transform = transforms.Compose([
#     transforms.Grayscale(1)
#     transforms.Resize(32),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# =============================================================================

  transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))])
  mold_img = ImageFolder('data/folder/train', transform)

  loader = data.DataLoader(mold_img, BATCH_SIZE, shuffle=True)

  noise = torch.randn(64, NLAT, 1, 1, device=device)

  EG_losses, C_losses = [], []
  curr_iter = C_iter = EG_iter = 0
  C_update, EG_update = True, False
  print('Training starts...')

  while curr_iter < ITER:
    for batch_idx, (x, _) in enumerate(loader, 1):
      x = x.to(device)

      if curr_iter == 0:
        init_x = x
        curr_iter += 1

      z = torch.randn(x.size(0), NLAT, 1, 1).to(device)
      C_loss, EG_loss = wali(x, z, lamb=LAMBDA)

      if C_update:
        optimizerC.zero_grad()
        C_loss.backward()
        C_losses.append(C_loss.item())
        optimizerC.step()
        C_iter += 1

        if C_iter == C_ITERS:
          C_iter = 0
          C_update, EG_update = False, True
        continue

      if EG_update:
        optimizerEG.zero_grad()
        EG_loss.backward()
        EG_losses.append(EG_loss.item())
        optimizerEG.step()
        EG_iter += 1

        if EG_iter == EG_ITERS:
          EG_iter = 0
          C_update, EG_update = True, False
          curr_iter += 1
        else:
          continue

      # print training statistics
      if curr_iter % 100 == 0:
        print('[%d/%d]\tW-distance: %.4f\tC-loss: %.4f'
          % (curr_iter, ITER, EG_loss.item(), C_loss.item()))

        # plot reconstructed images and samples
        wali.eval()
        real_x, rect_x = init_x[:32], wali.reconstruct(init_x[:32]).detach_()
        rect_imgs = torch.cat((real_x.unsqueeze(1), rect_x.unsqueeze(1)), dim=1)
        rect_imgs = rect_imgs.view(64, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).cpu()
        genr_imgs = wali.generate(noise).detach_().cpu()
        utils.save_image(rect_imgs * 0.5 + 0.5, 'Runs/rect%d.png' % curr_iter)
        utils.save_image(genr_imgs * 0.5 + 0.5, 'Runs/genr%d.png' % curr_iter)
        wali.train()

      # save model
      if curr_iter % (ITER // 10) == 0:
        torch.save(wali.state_dict(), 'Runs/%d.ckpt' % curr_iter)

#   # plot training loss curve
#   plt.plot(EG_losses, label='Encoder + Generator')
#   plt.plot(C_losses, label='Critic')
# # plt.figure(figsize=(10, 5))
#   plt.title('Training loss curve')
#   plt.xlabel('Iterations')
#   plt.ylabel('Loss')
#   plt.legend()
#   plt.savefig('loss_curve.png')


if __name__ == "__main__":
  main()
