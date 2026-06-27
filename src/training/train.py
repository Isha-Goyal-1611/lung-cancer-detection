import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def weighted_cross_entropy_loss(predictions, targets, pos_weight=99.0):
   weights = targets * pos_weight+(1-targets)*1.0
   loss_fn=nn.BCELoss(weight=weights)
   return loss_fn(predictions,targets)

def focal_loss(predictions, targets, gamma=2.0):
   bce_loss = nn.BCELoss(reduction='none')(predictions, targets)
   p_t = predictions * targets + (1 - predictions) * (1 - targets)
   focal_weight=(1-p_t)**gamma
   focal_loss=focal_weight*bce_loss
   return focal_loss.mean()

def run_fake_training():
   targets=torch.zeros(100)
   targets[0]=1.0

   predictions=torch.full((100,),0.05)
   standard_loss=nn.BCELoss()(predictions,targets)
   weighted_loss = weighted_cross_entropy_loss(predictions,targets)
   focal=focal_loss(predictions,targets)
   print(f"Standard BCE loss:    {standard_loss.item():.4f}")
   print(f"Weighted loss:        {weighted_loss.item():.4f}")
   print(f"Focal loss:           {focal.item():.4f}")
   print()
   print("Which loss would punish the model MORE for missing the nodule?")

if __name__=="__main__":
   run_fake_training()
 