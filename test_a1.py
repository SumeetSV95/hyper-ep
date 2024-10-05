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
import psutil
import scipy.io
def read_matrix(file_name):
    return np.fromfile(file_name,dtype=np.double) 



def generate_batch(start, end, batch_size,dimD):
    batch=torch.from_numpy(np.zeros((batch_size,dimD*2))).to(torch.float32).to(device)
    j=0
    print(start,end)
    for i in range(start, end):
        batch[j][i] = 0.5
        j+=1
    return batch


def get_memory_usage():
    """Get current memory usage of the process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # in bytes

class StepCountingModel(nn.Module):
    def __init__(self, base_model):
        super(StepCountingModel, self).__init__()
        self.base_model = base_model
        self.nfe = 0  # Number of function evaluations

    def forward(self, t, y):
        self.nfe += 1  # Increment the step counter each time the model is called
        return self.base_model(t, y)


def getPlotModel(S, H, start, end, dimD, batch_size, device, t, name_prefix):
    total_time = 0
    total_memory = 0
    total_steps = 0
    num_runs = 0

    fig_tmp, ax_tmp = plt.subplots()  # Create a single figure for TMP
    fig_egm, ax_egm = plt.subplots()  # Create a single figure for EGM

    for i in range(8, 16, 2):
        i = i / 100   
        par = torch.from_numpy(np.full((dimD, 1), i)).float().to(device)
        
        base_model = APModel(S, par, batch_size, dimD)
        model = StepCountingModel(base_model)  # Wrap the base model in step counting model
        model.to(device)
        
        # Generate batch
        batch = generate_batch(start, end, batch_size, dimD)
        
        # Track time and memory
        start_time = time.time()
        start_memory = get_memory_usage()

        # Solve TMP using ODE solver
        tmp = odeint(model, batch, t, method='dopri5')

        end_time = time.time()
        end_memory = get_memory_usage()
        
        elapsed_time = end_time - start_time
        memory_used = (end_memory - start_memory) / (1024 * 1024)  # Convert to MB
        
        total_time += elapsed_time
        total_memory += memory_used
        total_steps += model.nfe  # Add the number of function evaluations (steps)
        num_runs += 1
        
        TMP = tmp[:, :, 0:dimD].permute(1, 0, 2).cpu().numpy()
        EGM = torch.matmul(H, tmp[:, :, 0:dimD].permute(0, 2, 1)).permute(0, 2, 1).cpu().numpy()
        
        # Plot TMP on the same figure
        ax_tmp.plot(TMP[0, :, :], linewidth=0.5, label=f'param={i:.2f}')
        
        # Plot EGM on the same figure
        ax_egm.plot(EGM[:, 0, 1], linewidth=0.5, label=f'param={i:.2f}')
    
    # Finalizing the TMP plot
    ax_tmp.set_title('TMP Plot for all i values')
    ax_tmp.legend(loc='best')
    ax_tmp.set_xlabel('Time')
    ax_tmp.set_ylabel('TMP Value')
    fig_tmp.savefig(f'{name_prefix}_all_TMP.png')
    plt.close(fig_tmp)
    
    # Finalizing the EGM plot
    ax_egm.set_title('EGM Plot for all i values')
    ax_egm.legend(loc='best')
    ax_egm.set_xlabel('Time')
    ax_egm.set_ylabel('EGM Value')
    fig_egm.savefig(f'{name_prefix}_all_EGM.png')
    plt.close(fig_egm)
    
    # Calculate average time, memory usage, and steps
    avg_time = total_time / num_runs
    avg_memory = total_memory / num_runs
    avg_steps = total_steps / num_runs
    
    print(f'Average time for {name_prefix}: {avg_time:.2f} seconds')
    print(f'Average memory used for {name_prefix}: {avg_memory:.2f} MB')
    print(f'Average number of steps for {name_prefix}: {avg_steps:.2f} steps')
       
batch_size = 16    
uv_path = "/home/sv6234/ECGI/data/TMP_data_UV_new/1.3/"
data_path = "/home/sv6234/ECGI/data/TMP_data_GT_new/1.3/"
dimD = 1862
device = 'cuda:0' 
print(device)
#max = 0.16
mat = scipy.io.loadmat('S_Trans.mat')
mat = mat['S']
S=torch.from_numpy(mat).float().to(device)
mat = scipy.io.loadmat('H_Trans.mat')
mat = mat['H']
H=torch.from_numpy(mat).float().to(device)
mat = scipy.io.loadmat('S_Trans_new.mat')
mat = mat['S']
S1=torch.from_numpy(mat).float().to(device)
mat = scipy.io.loadmat('H_Trans_new.mat')
mat = mat['H']
H1=torch.from_numpy(mat).float().to(device)
t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
val=0
for i in range(375):
    t[i] = val
    val += 0.9
t1=torch.from_numpy(np.linspace(0,350,117)).to(device)
print(t1)


# 0.4 goes beyound gives wrong solution. 0.06 wrong sol. correct range(0.07 - 0.18)


getPlotModel(S1,H1,int(200),int(216),1119,16,device,t1,'a2.png')
getPlotModel(S,H,int(200),int(216),1862,16,device,t,'a1.png')


    
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
    
    
    