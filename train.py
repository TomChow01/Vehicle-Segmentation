# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:09:24 2021

@author: hp
"""
import os
import torch
from loss import DiceLoss
from UNet import Model
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(train_loader, test_loader, lr, epochs,
          model_weights_dir = None, save_checkpoint = True,
          pretrained_weights = None):
    
  if torch.cuda.is_available():
      device = torch.device('cuda:0')
      print("Training on GPU..")
  else:
      device = torch.device('cpu')
      print("Training on CPU..")

  net = Model().to(device)

  dice_loss = DiceLoss(smooth = 1.)
  optim = torch.optim.Adam(net.parameters(), lr = lr)

  epoch_loss = {'training loss' : [], 'validation loss': []}

  if pretrained_weights:
    print("Loading Pretrained weights...")
    net.load_state_dict(torch.load(pretrained_weights))

  for epoch in range(1 , epochs+1):

      # if epoch > 0 and epoch % 10 == 0:

      #   liveloss.update(logs)
      #   liveloss.send()

      net.train()
      running_loss = 0.
      validation_loss = 0.
      logs = {}

      for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader)):
          # Transfer to GPU
          X_batch = X_batch.to(device = device, dtype = torch.float32)
          y_batch = y_batch.to(device = device, dtype = torch.float32)
          #print(X_batch.size())
          optim.zero_grad()
          output = net(X_batch)
          #print(output.size(), y_batch.size())
          loss = dice_loss(output,y_batch)

          #print(loss)

          #print(output[0].item(), y_batch[0].item()) 

          loss.backward()
          optim.step()

          running_loss += loss
      epoch_loss['training loss'].append(running_loss/len(train_loader.dataset))
      print("Training loss for epoch %d is %.4f " %(epoch, running_loss/len(train_loader.dataset)))

      #test_model(weight_path, test_loader, n_outputs=3)
      if save_checkpoint:
        if epoch % 10 == 0:
            if not os.path.exists(model_weights_dir):
                os.makedirs(model_weights_dir)
            print("Saving model weights...")
            weight_path = model_weights_dir + 'weights_epoch_' + str(epoch) + '.pt'
            torch.save(net.state_dict(), weight_path)


      #Validation
      # net.eval()
      # y_true = []
      # y_pred = []
      # for X_batch, y_batch in test_loader:
      #   X_batch, y_batch = X_batch.to(device), y_batch.to(device)

      #   out = net(X_batch)
      #   loss = dice_loss(out, y_batch)
      #   validation_loss += loss

      #   # y_true.append(int(y_batch.item()))
      #   # y_pred.append(int(out.item()))

      # epoch_loss['validation loss'].append(validation_loss/len(test_loader.dataset))  
      # print("Validation loss for epoch %d is %.4f " %(epoch, validation_loss))

      # print("Actual Heartrate:    ", y_true[:20])
      # print("Predicted Heartrate: ", y_pred[:20])

      # logs['training_loss'] = training_loss
      # logs['validation_loss'] = validation_loss

    # liveloss.update(logs)
    # livesloss.send()
      if epoch % 2 == 0:
        plt.plot(epoch_loss['training loss'])
        plt.show()