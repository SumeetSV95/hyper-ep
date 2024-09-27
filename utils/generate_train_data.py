import torch
import os
import numpy as np



dimD = 1862
data_path = "/home/sv6234/ECGI/data/TMP_data_UV/"

device = 'cuda:0'
for j in range(0,dimD):
    
    UV0 = np.zeros((dimD*2,1))
    UV0[j] = 0.5
    UV0 =torch.from_numpy(UV0).float().to(device)
    UV = UV0.squeeze().unsqueeze(0)
    torch.save(UV,os.path.join(data_path,str(j)+".pt"))