#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:04:16 2020

@author: brad
"""
# prerequisites
import torch
import numpy as np
from sklearn import svm
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim

from mVAE import train, test, vae,  thecolorlabels, optimizer, dataset_builder, load_checkpoint

checkpoint_folder_path = f'output' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

bs=100
data_set_flag ='padded_mnist'
train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip = dataset_builder(data_set_flag, bs)

for epoch in range(1, 201):
    train(epoch,'iterated', train_loader_noSkip, train_loader_skip, test_loader_noSkip)
 
    if epoch % 5 == 0:
        test('all',test_loader_noSkip, test_loader_skip, bs)  
   
    if epoch in [1,25,50,75,100,150,200,300,400,500]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_threeloss_singlegrad{str(epoch)}.pth')






