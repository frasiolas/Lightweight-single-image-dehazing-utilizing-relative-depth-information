import torch
from PIL import Image
import torchvision
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageOps
from torchvision.transforms.functional import crop

def renormalize(tensor):
    minFrom= tensor.min()
    maxFrom= tensor.max()
    minTo = 0
    maxTo=1
    return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))
    
class RESIDE_train(Dataset):

  def __init__(self):
    self.data = pd.read_csv(r'/home/frasiolas/unified_model/csv/Reside_train.csv', sep = ',')
    self.n_samples = self.data.shape[0]


  def __getitem__(self, index):
    path_clear =  self.data._get_value(index, 'Clear', takeable=False)
    clear = Image.open(path_clear)
    path_hazy = self.data._get_value(index, 'Hazy', takeable=False)
    haze = Image.open(path_hazy)
    path_depth = self.data._get_value(index, 'Depth', takeable=False)
    depth = Image.open(path_depth)
    hazy_transforms = transforms.Compose([
       transforms.Resize((448,608)),
       transforms.ToTensor(),
       #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    #resize =  transforms.Resize((448,608))
    to_tensor = transforms.ToTensor()
    
    horizontal_flip = transforms.RandomHorizontalFlip(p=1)
    if random.random() > 0.5:
      clear = horizontal_flip(clear)
      haze = horizontal_flip(haze)
      depth = horizontal_flip(depth)

    #clear = hazy_transforms(clear)
    clear = to_tensor(clear)
    haze = hazy_transforms(haze)
    #depth = resize(depth)
    depth = to_tensor(depth)
    
   
  
    return {'Clear': clear, 'Hazy': haze, 'Depth':depth}

  def __len__(self):
    return self.n_samples


class RESIDE_SOTS(Dataset):

  def __init__(self):
    self.data = pd.read_csv(r'/home/frasiolas/unified_model/csv/Reside_SOTS.csv', sep = ',')
    self.n_samples = self.data.shape[0]


  def __getitem__(self, index):
    path_clear =  self.data._get_value(index, 'Clear', takeable=False)
    clear = Image.open(path_clear)
    path_hazy = self.data._get_value(index, 'Hazy', takeable=False)
    haze = Image.open(path_hazy)
    hazy_transforms = transforms.Compose([
       transforms.Resize((448,608)),
       transforms.ToTensor(),
    ])
 
    clear_GT_trans = transforms.Compose([
       transforms.ToTensor(),
  
    ])
    
    clear = clear_GT_trans(clear)
    haze = hazy_transforms(haze)
    
    return {'Clear': clear, 'Hazy': haze}

  def __len__(self):
    return self.n_samples
  

  
class RESIDE_train_OTS(Dataset):

  def __init__(self):
    self.data = pd.read_csv(r'D:\PanagiotisFrasiolas\RESCUER\unified_model\csv\RESIDE_OTS.csv', sep = ',')
    self.n_samples = self.data.shape[0]


  def __getitem__(self, index):
    path_clear = self.data._get_value(index, 'Clear', takeable=False)
    clear = Image.open(path_clear)
    clear = clear.convert("RGB")

    path_hazy = self.data._get_value(index, 'Hazy', takeable=False)
    haze = Image.open(path_hazy)
    haze = haze.convert("RGB")

    path_depth = self.data._get_value(index, 'Depth', takeable=False)
    depth = Image.open(path_depth)

    hazy_transforms = transforms.Compose([
       transforms.Resize((480,480)),
       transforms.ToTensor(),
       #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    #resize =  transforms.Resize((448,608))
    to_tensor = transforms.ToTensor()
    
    horizontal_flip = transforms.RandomHorizontalFlip(p=1)
    if random.random() > 0.5:
      clear = horizontal_flip(clear)
      haze = horizontal_flip(haze)
      depth = horizontal_flip(depth)

    clear = hazy_transforms(clear)
    haze = hazy_transforms(haze)
    depth = hazy_transforms(depth)
    depth = depth.mean(dim=0)
    depth = depth.unsqueeze(0) 
   
  
    return {'Clear': clear, 'Hazy': haze, 'Depth':depth}

  def __len__(self):
    return self.n_samples


class RESIDE_SOTS_out(Dataset):

  def __init__(self):
    self.data = pd.read_csv(r'D:\PanagiotisFrasiolas\RESCUER\unified_model\csv\RESIDE_SOTS_out.csv', sep = ',')
    self.n_samples = self.data.shape[0]

  def __getitem__(self, index):
    path_clear =  self.data._get_value(index, 'Clear', takeable=False)
    clear = Image.open(path_clear)
    path_hazy = self.data._get_value(index, 'Hazy', takeable=False)
    haze = Image.open(path_hazy)
    hazy_transforms = transforms.Compose([
       transforms.Resize((480,480)),
       transforms.ToTensor(),
    ])
 
    clear_GT_trans = transforms.Compose([
       transforms.ToTensor(),
  
    ])
    
    clear = hazy_transforms(clear)
    haze = hazy_transforms(haze)
    
    return {'Clear': clear, 'Hazy': haze}

  def __len__(self):
    return self.n_samples

