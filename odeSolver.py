from torchdiffeq import odeint
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time




class APModel(nn.Module):
    def __init__(self, S, par) -> None:
        super(APModel, self).__init__()
        self.S = S
        self.par = par
    def forward(self,t,y):
        u = y[0:dimD].reshape(dimD, 1)
        v = y[dimD:dimD*2].reshape(dimD, 1)
        k = 8
        e = 0.01
        MK = torch.matmul(self.S, u)
        #print((u*v).shape)
        pde1 = MK + k*u*(1-u)*(u-self.par) - u*v
        pde2 = -e*(k*u*(u-self.par-1)+v)
        return torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0)


if __name__=="__main__":
    dimD = 1862
    device = 'cuda:0'
    def read_matrix(file_name):
        return np.fromfile(file_name) 

    #S1 = torch.from_numpy(np.load('data/S.npy')).float().to(device)
    #par = torch.from_numpy(np.load('data/par.npy')).float().to(device)
    par = torch.from_numpy(np.full((1862,1),0.15)).float().to(device)
    UV0 = torch.from_numpy(np.load('data/UV0.npy')).float().to(device)
    
    #H = torch.from_numpy(np.load('data/H.npy')).float().to(device)
    a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
    S=torch.from_numpy(a.reshape((1862,1862))).float().to(device)
    model=APModel(S, par)
    t=torch.from_numpy(np.zeros(200)).to(device)
    val=0
    for i in range(200):
        t[i] = val
        val += 0.9
    for i in range (16):
        UV0 = np.zeros((dimD*2,1))
        UV0[i] = 0.5
        UV0=torch.from_numpy(UV0).float().to(device)
        t0 = time.time()
        y=odeint(model, UV0.reshape((3724)), t, method='dopri5')
        print('{} seconds'.format(time.time() - t0))
        y=y.to('cpu')
        TMP = y[:, :dimD]
        print(TMP.shape)
        plt.plot(TMP[:,:],linewidth=0.1)
        plt.savefig('testplot'+str(i)+'.png')
        plt.close()

