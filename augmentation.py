# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:16:25 2021

@author: hp
"""
import os
import random
import numpy as np
from tqdm import tqdm
import skimage as sk
from skimage import img_as_ubyte
from skimage import transform
from skimage import util
from skimage.io import imread, imsave


# Different types of Augmentation
def random_rotation(image_array, mask_array):
    # pick a random degree of rotation between 45% on the left and 45% on the right
    random_degree = random.uniform(-45, 45)
    aug_img = sk.transform.rotate(image_array, random_degree)
    aug_mask = sk.transform.rotate(mask_array, random_degree)
    return aug_img, aug_mask 

def random_noise(image_array, mask_array):
    # add random noise to the image
    return sk.util.random_noise(image_array), mask_array

def horizontal_flip(image_array, mask_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return np.fliplr(image_array), np.fliplr(mask_array)
    #return image_array[:, ::-1], mask[:, ::-1]

def vertical_flip(image_array, mask_array):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return np.flipud(image_array), np.flipud(mask_array)

def flip_and_rotate(image_array, mask_array):
    flip_img, flip_mask = np.fliplr(image_array), np.fliplr(mask_array)
    random_degree = random.uniform(-45, 45)
    aug_img = sk.transform.rotate(flip_img, random_degree)
    aug_mask = sk.transform.rotate(flip_mask, random_degree)
    return aug_img, aug_mask 

def augment_images(img_dir, mask_dir, aug_img_dir, aug_mask_dir):
  num_aug_imgs = 1
  for sample in tqdm(os.listdir(img_dir)):
  #for sample in os.listdir(img_dir):    
    #test = imread(os.path.join(img_dir, sample))
    img = imread(os.path.join(img_dir, sample))
    mask = imread(os.path.join(mask_dir,sample), 0)

    # aug_img_dir = os.path.join(main_dir, 'aug_img')
    # aug_mask_dir = os.path.join(main_dir, 'aug_mask')

    if not os.path.exists(aug_img_dir):
      os.makedirs(aug_img_dir)
    if not os.path.exists(aug_mask_dir):
      os.makedirs(aug_mask_dir)

    # Save the original image
    imsave(aug_img_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(img))
    imsave(aug_mask_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(mask))

    num_aug_imgs += 1

    # Save Rotated Images
    for i in range(5):
      aug_img, aug_mask = random_rotation(img, mask)

      imsave(aug_img_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_img))
      imsave(aug_mask_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_mask))

      num_aug_imgs += 1

    # Save Noisy Images
    aug_img, aug_mask = random_noise(img, mask)
    imsave(aug_img_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_img))
    imsave(aug_mask_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_mask))

    num_aug_imgs += 1

    # Save Horizontaly Flipped Images
    aug_img, aug_mask = horizontal_flip(img, mask)
    imsave(aug_img_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_img))
    imsave(aug_mask_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_mask))

    num_aug_imgs += 1

    # Save Vertically Flipped Images
    aug_img, aug_mask = vertical_flip(img, mask)
    imsave(aug_img_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_img))
    imsave(aug_mask_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_mask))

    num_aug_imgs += 1

    # Save Flipped and Rotated Images
    aug_img, aug_mask = flip_and_rotate(img, mask)
    imsave(aug_img_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_img))
    imsave(aug_mask_dir + '/{}.jpg'.format(num_aug_imgs), img_as_ubyte(aug_mask))

    num_aug_imgs += 1

  print("Total images after augmentation: ", num_aug_imgs-1)