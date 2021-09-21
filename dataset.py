# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:47:40 2021

@author: hp
"""
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class VehicleDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_paths, mask_paths, resize_to = 256):

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_paths = img_paths
        self.mask_paths = mask_paths

        self.resize_to = resize_to

    def __getitem__(self, index):

      img_path = self.img_paths[index]
      mask_path = self.mask_paths[index]

      # print(os.path.join(self.img_dir, img_path))
      # print(os.path.join(self.mask_dir,mask_path))

      img = cv2.imread(os.path.join(self.img_dir, img_path), 0)
      mask = cv2.imread(os.path.join(self.mask_dir,mask_path), 0)

      # Resize and Scale

      img = cv2.resize(img, (self.resize_to, self.resize_to)) / 255.
      mask = cv2.resize(mask, (self.resize_to, self.resize_to))



      # Convert Non Binary Mask to Binary Mask
      mask = (mask>127).astype(int)

      # Convert to Pytorch Tensor
      img = torch.tensor(img, dtype = torch.float32)
      mask = torch.tensor(mask, dtype = torch.uint8)

      # if self.transform_img:
      #   img = self.transform_img(img)
      #   mask = self.transform_mask(mask)
      #print(img.size(), mask.size())
      
      #return img.permute(2,0,1), mask.unsqueeze(0)
      return img.unsqueeze(0), mask.unsqueeze(0)
        
    
    def __len__(self):
        return len(self.img_paths)