# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:49:07 2021

@author: hp
"""
import torch
from UNet import Model
from loss import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_model(weight_path, test_loader, show_outputs = 10):
  device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
  net = Model().to(device)
  print("Loading Pretrained weights...")
  net.load_state_dict(torch.load(weight_path))
  net.eval()
  dice_loss = DiceLoss(smooth = 1.)
  validation_loss = 0.
  y_true = []
  y_pred = []
  c = 0
  batch = 0
  for X_batch, y_batch in tqdm(test_loader):
    batch += 1
    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

    out = net(X_batch)
    loss = dice_loss(out, y_batch.float())
    validation_loss += loss
    out = torch.sigmoid(out)

    #print(y_batch.size(), out.size())
    # loss = dice_loss(out, y_batch)
    # validation_loss += loss

    original_seg_map = y_batch.cpu().numpy()
    pred_seg_map = out.detach().cpu().numpy()

    #print(original_seg_map.shape, pred_seg_map.shape)

    for i in range(original_seg_map.shape[0]):
      y_true.append(original_seg_map[i][0])
      y_pred.append(pred_seg_map[i][0])
    
    if batch == 10:
      break

  if show_outputs is not None:
    print("Showing model results...")

    # print(np.unique(original_seg_map))
    # print(np.unique(pred_seg_map))

    for i in range(show_outputs):
      c += 1
      print("Image: ", c)
      width=5
      height=5
      rows = 1
      cols = 2
      axes=[]
      fig=plt.figure()

      for a in range(rows*cols):
        axes.append(fig.add_subplot(rows, cols, a+1) )
        if a == 0:
          subplot_title=("Original")
          axes[-1].set_title(subplot_title)  
          plt.imshow(y_true[i])
        else:
          subplot_title=("Predicted")
          axes[-1].set_title(subplot_title)  
          plt.imshow(y_pred[i])

      fig.tight_layout()    
      plt.show()

  print("Validation loss: %.4f " %(validation_loss / len(test_loader.dataset)))