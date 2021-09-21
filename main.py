# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:22:59 2021

@author: hp
"""
import os
from augmentation import *
from train import train_model
from test_model import test_model
from dataset import VehicleDataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold

main_dir = 'G:/ML Projects/Personal Proects/Veichle Segmentation (Intozi)/segmentation/'

img_dir = main_dir + 'img'
mask_dir = main_dir + 'mask' 

aug_img_dir = main_dir + 'aug_img'
aug_mask_dir = main_dir + 'aug_mask'

augment = False
img_size = 64
lr = 0.0001
batch_size = 8
n_epochs = 200
train = False
model_weights_dir = main_dir + 'model_weights/'
weight_path = model_weights_dir + 'weights_epoch_50.pt'

if __name__ == '__main__':

  if augment:
    print("Augmenting Images..")
    augment_images(img_dir, mask_dir, aug_img_dir, aug_mask_dir)

  aug_img_paths = sorted(os.listdir(aug_img_dir))
  aug_mask_paths = sorted(os.listdir(aug_mask_dir))

  X_train_paths, X_test_paths, y_train_paths, y_test_paths = train_test_split(aug_img_paths, aug_mask_paths, test_size = 0.1, random_state = 10)

  train_dataset = VehicleDataset(aug_img_dir, aug_mask_dir, X_train_paths, y_train_paths, resize_to = img_size)
  test_dataset = VehicleDataset(aug_img_dir, aug_mask_dir, X_test_paths, y_test_paths, resize_to = img_size)

  print("Total number of training data: ", len(train_dataset))
  print("Total number of test data: ", len(test_dataset))

  train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)

  test_loader = DataLoader(test_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=2)

  if train:
    print("Training Model...")
    train_model(train_loader = train_loader, test_loader = test_loader, lr = lr, epochs = n_epochs,
                  model_weights_dir = model_weights_dir, save_checkpoint = True, pretrained_weights = None)
  else:
    print("Testing Model Performance...")
    test_model(weight_path, test_loader, show_outputs=10)