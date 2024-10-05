from torchdiffeq import odeint
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io




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
        
        u = torch.t(y[:,0:self.dimD])
        
        v = torch.t(y[:,self.dimD:self.dimD*2])
       
        k = 8
        e = 0.01
        MK = torch.matmul(self.S, u)
        val = u*v
        pde1 = MK + k*u*(1-u)*(u-self.par) -u*v
        pde2 = -e*(k*u*(u-self.par-1)+v)
        if val.min()<self.val_min:
            self.val_min= val.min()
        if val.max()>self.val_max:
            self.val_max= val.max()
        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        return torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))
    




if __name__=="__main__":
    
    def generate_mat(tmp,output_path):
        #filePath = '/home/sv6234/ECGI/data/TMP_data_GT_new/0.pt'
        #tmp=torch.load(filePath)
        #tmp=tmp.numpy()
        print(tmp.shape)
        scipy.io.savemat(output_path, mdict={'U': tmp})
    dimD = 1862
    device = 'cuda:0'
    def read_matrix(file_name):
        return np.fromfile(file_name) 

    #S1 = torch.from_numpy(np.load('data/S.npy')).float().to(device)
    #par = torch.from_numpy(np.load('data/par.npy')).float().to(device)
    batch_size = 16
    par = torch.from_numpy(np.full((1862,batch_size),0.15)).to(torch.float32).to(device)
    UV0 = torch.from_numpy(np.load('data/UV0.npy')).float().to(torch.float32).to(device)
    batch=torch.from_numpy(np.zeros((batch_size,dimD*2))).to(torch.float32).to(device)
    j = 0
    for i in range(batch_size):
        batch[i][j] = 0.5
        j+=1
    print(batch)
    #H = torch.from_numpy(np.load('data/H.npy')).float().to(device)
    a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
    S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float32).to(device)
    
    mat = scipy.io.loadmat('/home/sv6234/ECGI/S_Trans.mat')
    mat = mat['S']
    S1=torch.from_numpy(mat).float().to(device)
    model=APModel(S, par,batch_size,dimD)
    model1 = APModel(S1, par,batch_size,dimD)
    t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
    val=0
    for i in range(375):
        t[i] = val
        val += 0.9
    t0 = time.time()
    y=odeint(model, batch, t, method='dopri5')
    y1=odeint(model1, batch, t, method='dopri5')
    print('{} seconds'.format(time.time() - t0))
    y=y.to('cpu')
    TMP = y[:, :, 0:dimD].permute(1,0,2).to('cpu').detach().numpy()
    TMP_1 = y1[:, :, 0:dimD].permute(1,0,2).to('cpu').detach().numpy()    
    print(TMP.shape)
    for i in range(4):
        generate_mat(TMP[i,:,:], 'gt_out_old'+str(i)+'.mat')
        generate_mat(TMP_1[i,:,:], 'gt_out_new'+str(i)+'.mat')
        
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, figsize=(24, 3))
    ax1.plot(TMP[0,:,:],linewidth=0.1)
    ax2.plot(TMP_1[0,:,:],linewidth=0.1)
    ax3.plot(TMP[1,:,:],linewidth=0.1)
    ax4.plot(TMP_1[1,:,:],linewidth=0.1)
    plt.savefig('gt_compare.png')
    
    #nums=np.random.randint(0, 1862,1)
    #mat = scipy.io.loadmat('/home/sv6234/ECGI/TmpSeg0exc3.mat')
    #print(nums)
    
    #mse=(np.square(TMP[:,0,nums[0]] - mat["U"][1:,nums[0]+1])).mean(axis=0)
    
    #print(mse)
    
        
    
    
