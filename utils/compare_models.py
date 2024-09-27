from torch.utils.data import DataLoader
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import HybridModel
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import scipy.io
import random
from utils import *
from dataset import TMPDataset


model1_name = 'model_EGM'
model2_name ='full'
model3_name='sur_in'
model4_name='sur_out'
batch_size = 16
device = 'cuda:0'
path = '/home/sv6234/ECGI/results/compare_all_1/'
create_folder_if_not_exists(path)
create_folder_if_not_exists(path+"internal/")
create_folder_if_not_exists(path+"surface/")
convert_to_EGM_model1 = True
convert_to_EGM_model2 = False
convert_to_EGM_model3 = False
convert_to_EGM_model4 = False

# Load the transformation matrices
mat = scipy.io.loadmat('/home/sv6234/ECGI/S_Trans.mat')
S = torch.from_numpy(mat['S']).float().to(device)
mat = scipy.io.loadmat('/home/sv6234/ECGI/H_Trans.mat')
H = torch.from_numpy(mat['H']).float().to(device)

# Load the models and set them to evaluation mode
model1 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkHybrid_EGM_dilate/best-model-epoch=10-val_loss=0.012.ckpt").to(device)
model1.eval()

model2 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkHybrid_TMP_Full_seed_1250/best-model-epoch=09-val_loss=0.011.ckpt").to(device)
model2.eval()

model3 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkHybrid_TMP_surface_in/best-model-epoch=11-val_loss=0.011.ckpt").to(device)
model3.eval()

model4 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkHybrid_TMP_sur_in_sur_out_1/best-model-epoch=08-val_loss=0.012.ckpt").to(device)
model4.eval()
# Load the test data
dataset=TMPDataset("/home/sv6234/ECGI/data/TMP_data_UV_new/","/home/sv6234/ECGI/data/TMP_data_GT_new/",15,H=None)
train_set, valid_set, test_set=torch.utils.data.random_split(dataset,[1056,352,352])
train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
def get_split_tmp(tmp, surfaceIds,internal_nodes):
    TMP_surface = tmp.permute(0,2,1)
    TMP_surface = TMP_surface[:,surfaceIds,:].squeeze().permute(0,2,1)
    TMP_internal = tmp.permute(0,2,1)
    TMP_internal = TMP_internal[:,internal_nodes,:].squeeze().permute(0,2,1)
    return TMP_surface, TMP_internal
for epoch in range(1):
    uv, y, k, a = next(iter(train_loader))
    
    with torch.no_grad():
        # Disable gradient calculation
        internal_nodes = []
        for i in range(1862):
             if i not in model1.surfaceIds:
                 internal_nodes.append(i)  
        torch.tensor(internal_nodes).to(device)
        y_surface, y_internal = get_split_tmp(y, model1.surfaceIds, internal_nodes)
        if convert_to_EGM_model1 or convert_to_EGM_model2 or convert_to_EGM_model3 or convert_to_EGM_model4:
            EGM_K = torch.zeros((batch_size, 15, 375, 396)).to(device)
            for i in range(batch_size):
                for j in range(15):
                    tmp = k[i, j, :, :]
                    #print(tmp.shape)
                    EGM = torch.matmul(H, tmp.permute(1, 0)).permute(1, 0)
                    EGM_K[i, j, :, :] = EGM
        if model1: 
            if convert_to_EGM_model1:
                tmp1 = model1(uv, EGM_K)
            else:
                tmp1 = model1(uv, k)
            surface_tmp_1,internal_tmp_1=get_split_tmp(tmp1, model1.surfaceIds, internal_nodes)
        if model2:
            if convert_to_EGM_model2:
                tmp2 = model2(uv, EGM_K)
            else:
                tmp2 = model2(uv, k)
            surface_tmp_2,internal_tmp_2=get_split_tmp(tmp2, model2.surfaceIds, internal_nodes)
        if model3:
            if convert_to_EGM_model3:
                tmp3 = model3(uv, EGM_K)
            else:
                tmp3 = model3(uv, k)
            surface_tmp_3,internal_tmp_3=get_split_tmp(tmp3, model3.surfaceIds, internal_nodes)
        if model4:
            if convert_to_EGM_model4:
                tmp4 = model4(uv, EGM_K)
            else:
                tmp4 = model4(uv, k)
            surface_tmp_4,internal_tmp_4=get_split_tmp(tmp4, model4.surfaceIds, internal_nodes)

        for i in range(batch_size):
            for j in random.sample(range(0, 1862), 2):
                fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
                ax1.plot(y[i, :, j].detach().cpu().numpy(), label='a gt' + str(a[i].item()))
                if model1:
                    ax1.plot(tmp1[i, :, j].detach().cpu().numpy(), label='a '+model1_name + str(model1.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model2:
                    ax1.plot(tmp2[i, :, j].detach().cpu().numpy(), label='a '+model2_name + str(model2.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model3:
                    ax1.plot(tmp3[i, :, j].detach().cpu().numpy(), label='a '+model3_name + str(model3.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model4:
                    ax1.plot(tmp4[i, :, j].detach().cpu().numpy(), label='a '+model4_name + str(model4.ode.par[i].detach().cpu().item()), linewidth=0.5)
                ax1.legend()
                plt.savefig(path + str(round(a[i].item(), 2)) + '_' + str(j) + '.png')
                plt.close()
        for i in range(batch_size):
            for j in random.sample(range(len(internal_nodes)),2):
                fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
                ax1.plot(y_internal[i, :, j].detach().cpu().numpy(), label='a gt' + str(a[i].item()))
                if model1:
                    ax1.plot(internal_tmp_1[i, :, j].detach().cpu().numpy(), label='a '+model1_name + str(model1.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model2:
                    ax1.plot(internal_tmp_2[i, :, j].detach().cpu().numpy(), label='a '+model2_name + str(model2.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model3:
                    ax1.plot(internal_tmp_3[i, :, j].detach().cpu().numpy(), label='a '+model3_name + str(model3.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model4:
                    ax1.plot(internal_tmp_4[i, :, j].detach().cpu().numpy(), label='a '+model4_name + str(model4.ode.par[i].detach().cpu().item()), linewidth=0.5)
                ax1.legend()
                plt.savefig(path+"internal/" + str(round(a[i].item(), 2)) + '_' + str(j) + '.png')
                plt.close()
                
        for i in range(batch_size):
            for j in random.sample(range(len(model1.surfaceIds)),2):
                fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
                ax1.plot(y_surface[i, :, j].detach().cpu().numpy(), label='a gt' + str(a[i].item()))
                if model1:
                    ax1.plot(surface_tmp_1[i, :, j].detach().cpu().numpy(), label='a '+model1_name + str(model1.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model2:
                    ax1.plot(surface_tmp_2[i, :, j].detach().cpu().numpy(), label='a '+model2_name + str(model2.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model3:
                    ax1.plot(surface_tmp_3[i, :, j].detach().cpu().numpy(), label='a '+model3_name + str(model3.ode.par[i].detach().cpu().item()), linewidth=0.5)
                if model4:
                    ax1.plot(surface_tmp_4[i, :, j].detach().cpu().numpy(), label='a '+model4_name + str(model4.ode.par[i].detach().cpu().item()), linewidth=0.5)
                ax1.legend()
                plt.savefig(path+"surface/" + str(round(a[i].item(), 2)) + '_' + str(j) + '.png')
                plt.close()
    a_dict = set()
    for i in range(16):
        if len(a_dict) == 4:
            break
        if a[i].item() not in a_dict:
            if round(a[i].item(),2) == 0.14:
                if y[i, :, 0].max() > 0.5:
                    a_dict.add(a[i].item())
                else:
                    continue
            else:
                a_dict.add(a[i].item())
    #generate_mat(tmp1[i,:,:], 'model_out_sur'+str(i)+'.mat')
            if epoch==0:
                generate_mat(tmp1[i,:,:].detach().cpu().numpy(), model1_name+str(round(a[i].item(),2))+'.mat')
                generate_mat(tmp2[i,:,:].detach().cpu().numpy(), model2_name+str(round(a[i].item(),2))+'.mat')
                generate_mat(tmp3[i,:,:].detach().cpu().numpy(), model3_name+str(round(a[i].item(),2))+'.mat')
                generate_mat(tmp4[i,:,:].detach().cpu().numpy(), model4_name+str(round(a[i].item(),2))+'.mat')
                generate_mat(y[i,:,:].detach().cpu().numpy(), 'gt_out'+str(round(a[i].item(),2))+'.mat')
            
