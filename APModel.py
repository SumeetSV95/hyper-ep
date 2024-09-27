import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from torchdiffeq import odeint
import time
import scipy.io
def read_matrix(file_name):
    return np.fromfile(file_name) 
dimD = 1862
device = 'cuda:0'
S1 = torch.from_numpy(np.load('data/S.npy')).float().to(device)
par = torch.from_numpy(np.load('data/par.npy')).float().to(device)
par = torch.from_numpy(np.full((1862,1),0.15)).float().to(device)
UV0 = torch.from_numpy(np.load('data/UV0.npy')).float().to(device)
H = torch.from_numpy(np.load('data/H.npy')).float().to(device)
mat = scipy.io.loadmat('/home/sv6234/ECGI/S_Trans.mat')
mat = mat['S']
T=torch.from_numpy(mat).float().to(device)
print(type(mat))
"""a=read_matrix('/home/sv6234/ECGI/Trans_state.bin')
S=torch.from_numpy(a.reshape((1862,1862))).float().to(device)"""
filename = '/home/sv6234/ECGI/Trans_state.bin'
with open(filename, 'rb') as fid:
    S = torch.frombuffer(fid.read(), dtype=torch.float64)
    S = S.view(dimD, -1).to(device)
print(S1 - T)
print(S1.shape, S.shape)


class APModel(nn.Module):
    def __init__(self, S, par, batch_size, dimD) -> None:
        super(APModel, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.val_max = -float('inf')
        self.val_min = float('inf')
    def forward(self,t,y):
        u = y[0:dimD].reshape(dimD, 1)
        v = y[dimD:dimD*2].reshape(dimD, 1)
        k = 8
        e = 0.01
        
        MK = torch.matmul(self.S, u)
        pde1 = MK + k*u*(1-u)*(u-self.par) - u*v
        pde2 = -e*(k*u*(u-self.par-1)+v)
        return torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0)


def RK5(y, h, fun):
    k1 = fun(y)
    k2 = fun(y + h*k1/4)
    k3 = fun(y + h*k1*3/32 + h*k2*9/32)
    k4 = fun(y + h*k1*1932/2197 + h*k2*-7200/2197 + h*k3*7296/2197)
    k5 = fun(y + h*k1*439/216 + h*k2*-8 + h*k3*3680/513 + h*k4*-845/4104)
    k6 = fun(y + h*k1*-8/27 + h*k2*2 + h*k3*-3544/2565 + h*k4*1859/4104 + h*k5*-11/40)
    dy = h * (k1*16/135 + k2*0 + k3*6656/12825 + k4*28561/56430 + k5*-9/50 + k6*2/55)
    return dy

def generate_batch(start, end, batch_size):
    batch=torch.from_numpy(np.zeros((batch_size,dimD*2))).to(torch.float32).to(device)
    j=0
    for i in range(start, end):
        batch[j][i] = 0.5
        j+=1
    return batch

t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
val=0
for i in range(375):
    t[i] = val
    val += 0.9


batch=generate_batch(1718,1734,16)
APModel = APModel(T, 0.1, 1, dimD)
y=odeint(APModel, batch[0], t, method='dopri5')
TMP = y[ :, 0:dimD]
print(TMP.shape)

plt.plot(TMP[:,:].cpu().numpy(),linewidth=0.5)
plt.savefig('test4.png')
print("done")
t0 = time.time()
UV = UV0.squeeze().unsqueeze(0)
for i in range(6000):
    dUV = RK5(UV[-1], 0.03, APModel)
    UV_ = UV[-1] + dUV
    UV = torch.cat((UV, UV_.unsqueeze(0)), dim=0)
TMP = UV[:, :dimD]
print('{} seconds'.format(time.time() - t0))
print(TMP.shape)
TMP=TMP.to('cpu')
for i in range (1862):
    plt.plot(TMP[:,i])
plt.savefig('test1.png')
"""BSP = torch.matmul(H, TMP.T).T
    
BSP = torch.matmul(H, TMP.T).T"""

