from torchdiffeq import odeint_adjoint as odeint
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time




class APModel():
    def __init__(self, S, par, batch_size, dimD) -> None:
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
    def forward(self,y):
        
        u = torch.t(y[:,0:dimD])
        
        v = torch.t( y[:,dimD:dimD*2])
       
        k = 8
        e = 0.01
        MK = torch.matmul(self.S, u)#(1862*1862)*(1862*bs)
        
        
        
        #print(u, 1- self.par)
        pde1 = MK + k*u*(1-u)*(u-self.par) - u*v #(1862*bs .* 1862*bs)
        pde2 = -e*(k*u*(u-self.par-1)+v)
        
        return torch.t(torch.cat((pde1, pde2), dim=0))
    
    def RK5(self, y, h, fun):
        k1 = fun(y)
        k2 = fun(y + h*k1/4)
        k3 = fun(y + h*k1*3/32 + h*k2*9/32)
        k4 = fun(y + h*k1*1932/2197 + h*k2*-7200/2197 + h*k3*7296/2197)
        k5 = fun(y + h*k1*439/216 + h*k2*-8 + h*k3*3680/513 + h*k4*-845/4104)
        k6 = fun(y + h*k1*-8/27 + h*k2*2 + h*k3*-3544/2565 + h*k4*1859/4104 + h*k5*-11/40)
        dy = h * (k1*16/135 + k2*0 + k3*6656/12825 + k4*28561/56430 + k5*-9/50 + k6*2/55)
        if torch.isnan(dy).any():
            print("in here")
        
        return dy
    def solve(self,UV):
        
        for i in range(6000):
            
            dUV = self.RK5(UV[:,-1], 0.03, self.forward)
            UV_ = UV[:,-1] + dUV
            prev=UV[:,-1]
            #print(UV_.unsqueeze(1).shape)
            UV = torch.cat((UV, UV_.unsqueeze(1)), dim=1)
            new=UV[:,-1]
            #print(prev==new)
            #print(UV.shape)
        return UV
    


if __name__=="__main__":
    dimD = 1862
    device = 'cuda:0'
    def read_matrix(file_name):
        return np.fromfile(file_name) 

    #S1 = torch.from_numpy(np.load('data/S.npy')).float().to(device)
    #par = torch.from_numpy(np.load('data/par.npy')).float().to(device)
    batch_size = 16
    par = torch.from_numpy(np.full((1862,1),0.15)).to(torch.float64).to(device)
    UV0 = torch.from_numpy(np.load('data/UV0.npy')).float().to(torch.float64).to(device)
    batch=torch.from_numpy(np.zeros((batch_size,dimD*2))).to(torch.float64).to(device)
    j = 0
    for i in range(batch_size):
        batch[i][j] = 0.5
        j+=1
    print(batch)
    #H = torch.from_numpy(np.load('data/H.npy')).float().to(device)
    a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
    S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float64).to(device)
    model=APModel(S, par,batch_size, dimD)
    
    batch = batch.squeeze().unsqueeze(0)
    batch=batch.reshape(batch_size, 1, 2*dimD)
    model=APModel(S,par, batch_size, dimD)
    t0 = time.time()
    UV=model.solve(batch)
    print('{} seconds'.format(time.time() - t0))
    UV=UV.to('cpu')
    TMP = UV[:,:,0:dimD]
    print(TMP[1,:,1].shape)
    plt.plot(TMP[1,:,1])
    plt.savefig('batch.png')
