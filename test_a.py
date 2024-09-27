import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot  as plt
import time

#from APModel import APModel, RK5
from batchOdeSolver import APModel
from torchdiffeq  import odeint
import os

def read_matrix(file_name):
    return np.fromfile(file_name) 



def generate_batch(start, end, batch_size):
    batch=torch.from_numpy(np.zeros((batch_size,dimD*2))).to(torch.float32).to(device)
    j=0
    for i in range(start, end):
        batch[j][i] = 0.5
        j+=1
    return batch
        
batch_size = 16    
uv_path = "/home/sv6234/ECGI/data/TMP_data_UV_new/1.3/"
data_path = "/home/sv6234/ECGI/data/TMP_data_GT_new/1.3/"
dimD = 1862
device = 'cuda:0'
#max = 0.16

t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
val=0
for i in range(375):
    t[i] = val
    val += 0.9


# 0.4 goes beyound gives wrong solution. 
for i in range(8,18,2):
    i = i/100   
    par = torch.from_numpy(np.full((1862,1),i)).float().to(device)
    a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
    S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float32).to(device)
    model=APModel(S, par,batch_size,dimD)
    print(i)
    model.to(device)
    end = i+batch_size
    batch=generate_batch(0,16,16)
    t0 = time.time()
    print(torch.cuda.memory_allocated(0))
    y=odeint(model, batch, t, method='scipy_solver',options={'solver':'RK45'})
    print(torch.cuda.memory_allocated(0))
    print('{} seconds'.format(time.time() - t0))  
    y=y.to('cpu')
    TMP = y[:, :, 0:dimD]
    TMP=TMP.permute(1,0,2)
    print(TMP.shape)
    index = 0
    print(model.val_min)
    print(model.val_max)

    
    print(TMP.shape)
    plt.plot(TMP[0,:,0])
    plt.savefig('test1.png')

"""UV=torch.load(uv_path+str(31)+".pt")    
print(torch.where(UV>0))
print(UV.shape)"""



    
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
    
    
    