import torch
import numpy as np
import scipy.io
filePath = '/home/sv6234/ECGI/data/TMP_data_GT_new/0.pt'
tmp=torch.load(filePath)
tmp=tmp.numpy()
print(tmp.shape)
scipy.io.savemat('out.mat', mdict={'U': tmp})