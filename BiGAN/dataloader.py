# -*- coding: utf-8 -*-
"""
Dataloader
"""

import os
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class Data:
    """ 
    Dataloader containing train and validation sets.
    """
    def __init__(self, train, valid):
        self.train = train
        self.valid = valid

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser


    Returns:
        [type]: dataloader
    """


    transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    train_ds = ImageFolder(os.path.join(opt.dataroot, 'train'), transform)
    valid_ds = ImageFolder(os.path.join(opt.dataroot, 'test'), transform)

    ## DATALOADER
    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batchsize, shuffle=False, drop_last=False)

    return Data(train_dl, valid_dl)
