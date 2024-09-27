from model import HybridModel
import torch
from dataset import TMPDataset
from sklearn.neighbors import kneighbors_graph
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.font_manager as font_manager
from lightning.pytorch import Trainer

train_loader=torch.load('/home/sv6234/ECGI/train_loader_phy.pt')
conter_dict  = {}
for i in range(66):
    print(i)
    
    
    uv, label, k, a = next(iter(train_loader))
    print(a)
    for val in a:
        if val.item() not in conter_dict:
            conter_dict[val.item()] = 1
        else:
            conter_dict[val.item()] +=1

    print(conter_dict)