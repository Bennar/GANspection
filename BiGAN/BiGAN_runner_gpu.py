# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:50:47 2020

@author: lazar
"""
import torch
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
ITER = 3000
IMAGE_SIZE = 256
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

# Defining the size of the CNN part of the network

layer_params = [[8,8,4,4,2,2],
                [4,4,2,2,2,2],
                [2,2,1,1,0,0]]

num_channels = [NUM_CHANNELS, 32, 64, 128, 256, 512, NLAT ]

def create_encoder():
  mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, num_channels[1], layer_params[0][0], layer_params[1][0], layer_params[2][0], bias=False), BatchNorm2d(num_channels[1]), ReLU(inplace=True),
    Conv2d(num_channels[1],num_channels[2], layer_params[0][1], layer_params[1][1], layer_params[2][1], bias=False), BatchNorm2d(num_channels[2]), ReLU(inplace=True),
    Conv2d(num_channels[2],num_channels[3], layer_params[0][2], layer_params[1][2], layer_params[2][2], bias=False), BatchNorm2d(num_channels[3]), ReLU(inplace=True),
    Conv2d(num_channels[3],num_channels[4], layer_params[0][3], layer_params[1][3], layer_params[2][3], bias=False), BatchNorm2d(num_channels[4]), ReLU(inplace=True),
    Conv2d(num_channels[4],num_channels[5], layer_params[0][4], layer_params[1][4], layer_params[2][4], bias=False), BatchNorm2d(num_channels[5]), ReLU(inplace=True),
    Conv2d(num_channels[5],num_channels[6], layer_params[0][5], layer_params[1][5], layer_params[2][5]))
  return DeterministicConditional(mapping)


def create_generator():
  mapping = nn.Sequential(
    ConvTranspose2d(num_channels[6],num_channels[5], layer_params[0][5], layer_params[1][5], layer_params[2][5], bias=False), BatchNorm2d(num_channels[5]), ReLU(inplace=True),
    ConvTranspose2d(num_channels[5],num_channels[4], layer_params[0][4], layer_params[1][4], layer_params[2][4], bias=False), BatchNorm2d(num_channels[4]), ReLU(inplace=True),
    ConvTranspose2d(num_channels[4],num_channels[3], layer_params[0][3], layer_params[1][3], layer_params[2][3], bias=False), BatchNorm2d(num_channels[3]), ReLU(inplace=True),
    ConvTranspose2d(num_channels[3],num_channels[2],  layer_params[0][2], layer_params[1][2], layer_params[2][2], bias=False), BatchNorm2d(num_channels[2]), ReLU(inplace=True),
    ConvTranspose2d(num_channels[2],num_channels[1],  layer_params[0][1], layer_params[1][1], layer_params[2][1], bias=False), BatchNorm2d(num_channels[1]), ReLU(inplace=True),
    ConvTranspose2d(num_channels[1],num_channels[0],  layer_params[0][0], layer_params[1][0], layer_params[2][0], bias=False), Tanh())
  return DeterministicConditional(mapping)


def create_critic():
  x_mapping = nn.Sequential(
    Conv2d(NUM_CHANNELS, num_channels[1], layer_params[0][0], layer_params[1][0], layer_params[2][0]), LeakyReLU(LEAK),
    Conv2d(num_channels[1], num_channels[2], layer_params[0][1], layer_params[1][1], layer_params[2][1]), LeakyReLU(LEAK),
    Conv2d(num_channels[2], num_channels[3], layer_params[0][2], layer_params[1][2], layer_params[2][2]),LeakyReLU(LEAK),
    Conv2d(num_channels[3], num_channels[4], layer_params[0][3], layer_params[1][3], layer_params[2][3]), LeakyReLU(LEAK),
    Conv2d(num_channels[4], num_channels[5], layer_params[0][4], layer_params[1][4], layer_params[2][4]), LeakyReLU(LEAK),
    Conv2d(num_channels[5], num_channels[6], layer_params[0][5], layer_params[1][5], layer_params[2][5]), LeakyReLU(LEAK))

  z_mapping = nn.Sequential(
    Conv2d(NLAT, 512, 1, 1, 0), LeakyReLU(LEAK),
    Conv2d(512, 512, 1, 1, 0), LeakyReLU(LEAK))

  joint_mapping = nn.Sequential(
    Conv2d(num_channels[6] + 512, 1024, 1, 1, 0), LeakyReLU(LEAK),
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
        utils.save_image(rect_imgs * 0.5 + 0.5, 'Runs/fsize/rect%d.png' % curr_iter)
        utils.save_image(genr_imgs * 0.5 + 0.5, 'Runs/fsize/genr%d.png' % curr_iter)
        wali.train()

      # save model
      if curr_iter % (ITER // 10) == 0:
        torch.save(wali.state_dict(), 'Runs/fsize/%d.ckpt' % curr_iter)

  # plot training loss curve
# =============================================================================
#   plt.figure(figsize=(10, 5))
#   plt.title('Training loss curve')
#   plt.plot(EG_losses, label='Encoder + Generator')
#   plt.plot(C_losses, label='Critic')
#   plt.xlabel('Iterations')
#   plt.ylabel('Loss')
#   plt.legend()
#   plt.savefig('loss_curve.png')
# =============================================================================


if __name__ == "__main__":
  main()
