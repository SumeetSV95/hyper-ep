import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot  as plt
import time

#from APModel import APModel, RK5
from batchOdeSolver import APModel
from torchdiffeq  import odeint
import os

import torch
import torch.nn as nn
import torchode as to
from TorchDiffEqPack import odesolve_adjoint
import scipy.io



device='cpu'
dimD = 1862
def read_matrix(file_name):
    return np.fromfile(file_name,dtype=np.double) 
mat = scipy.io.loadmat('H_Trans_new.mat')
mat = mat['H']
H=torch.from_numpy(mat).float().to(device)
#a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
#S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float32).to(device)
#S1 = torch.from_numpy(np.load('data/S.npy')).float().to(device)
data_path = '/home/sv6234/hyper-ep/data/TMP_data_GT_new/0.08/'

i=np.random.randint(0, 1119, 1)
print(i)
#path = os.path.join(data_path, str(i[0])+".pt")
path = os.path.join(data_path, str(17)+".pt")
tmp = torch.load(path).to(device)
#tmp=tmp.double()
egm = torch.matmul(H, tmp.permute(1,0)).permute(1,0)
plt.plot(egm[:,0].cpu().numpy())
plt.savefig('testing.png')
plt.close()
plt.plot(tmp[:,:].cpu().numpy(),linewidth=0.5)
plt.savefig('testing2.png')
plt.close()
"""nums=np.random.randint(0, 1862,1)
mat = scipy.io.loadmat('/home/sv6234/ECGI/TmpSeg0exc1719.mat')
#print(mat.shape)
print(tmp.shape)  
mse=(np.square(tmp[:,:].cpu().numpy() - mat["U"][1:,:])).mean(axis=0)
print(mse.shape)
print(mse)
plt.plot(tmp[:,:].cpu().numpy(),linewidth=0.5)
plt.savefig('testing3.png')
plt.close()
plt.plot(mat["U"][1:,:],linewidth=0.5)
plt.savefig('testing4.png')"""
