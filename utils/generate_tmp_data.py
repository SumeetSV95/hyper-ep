import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot  as plt
import time

#from APModel import APModel, RK5
from batchOdeSolver import APModel
from torchdiffeq import odeint
import os
from pathlib import Path
import scipy.io




def generate_batch(start, end, batch_size):
    batch=torch.from_numpy(np.zeros((batch_size,dimD*2))).to(torch.float32).to(device)
    j=0
    for i in range(start, end):
        batch[j][i] = 0.5
        j+=1
    return batch
        
batch_size = 32    
uv_path = "/home/sv6234/ECGI/data/TMP_data_UV_new/1.3/"
data_path = "/home/sv6234/ECGI/data/TMP_data_GT_new/1.3/"
dimD = 1862
device = 'cuda:0'
#max = 0.16

mat = scipy.io.loadmat('/home/sv6234/ECGI/S_Trans.mat')
mat = mat['S']
S=torch.from_numpy(mat).float().to(device)
t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
val=0
for i in range(375):
    t[i] = val
    val += 0.9


for k in range(8,18,2):
#for k in range(0,1):
    print("this is k")
    print(k)
    k = k/100
    folder_path = "/home/sv6234/ECGI/data/TMP_data_UV_new/"+str(k)+"/"
    os.makedirs(folder_path, exist_ok=True)
    folder_path="/home/sv6234/ECGI/data/TMP_data_GT_new/"+str(k)+"/"
    os.makedirs(folder_path, exist_ok=True)
    uv_path = "/home/sv6234/ECGI/data/TMP_data_UV_new/"+str(k)+"/"
    data_path = "/home/sv6234/ECGI/data/TMP_data_GT_new/"+str(k)+"/"
    if not os.path.exists(uv_path):
        Path(uv_path).mkdir()
        Path(data_path).mkdir()    
    par = torch.from_numpy(np.full((1862,1),k)).float().to(device)
    #par = torch.from_numpy(np.full((1862,1),0.15)).float().to(device)
    model=APModel(S, par,batch_size,dimD)
    for i in range(0,1856,batch_size):
    #for i in range(0,64,batch_size):
        print(i)
        start = i
        end = i+batch_size
        batch=generate_batch(start,end,batch_size)
        t0 = time.time()
        y=odeint(model, batch, t, method='dopri5')
        print('{} seconds'.format(time.time() - t0))  
        y=y.to('cpu')
        TMP = y[:, :, 0:dimD]
        TMP=TMP.permute(1,0,2)
        print(TMP.shape)
        index = 0
        for j in range(start,end):
            torch.save(batch[index],os.path.join(uv_path,str(j)+".pt"))
            torch.save(TMP[index],os.path.join(data_path,str(j)+".pt"))
            index += 1

        """TMP=torch.load(data_path+str(15)+".pt")
        print(TMP.shape)
        plt.plot(TMP[:,:],linewidth=0.1)
        plt.savefig('test.png')"""





    
"""for j in range(0,dimD,1):
    
UV0 = np.zeros((dimD*2,1))
UV0[j] = 0.5
UV0 =torch.from_numpy(UV0).float().to(device)
UV = UV0.squeeze().unsqueeze(0)
for i in range(6000):
    dUV = RK5(UV[-1], 0.03, APModel,S,dimD, par)
    UV_ = UV[-1] + dUV
    UV = torch.cat((UV, UV_.unsqueeze(0)), dim=0)
TMP = UV[:, :dimD]"""
"""TMP=TMP.to('cpu')
plt.plot(TMP[:,0])
plt.savefig('test2.png')"""
    #torch.save(TMP,os.path.join(data_path,str(j)+".pt"))
    
    
    