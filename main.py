from model import HybridModel, HybridModel1, OnlyNN, OnlyGCN
from lightning.pytorch import Trainer
from dataset import TMPDataset, TMPDatasetEQ
import torch
from torch.utils.data import DataLoader
import numpy as np
from torch.multiprocessing import Pool, Process, set_start_method
import torch.multiprocessing as mp 
from lightning.pytorch.callbacks import TQDMProgressBar
from sklearn.neighbors import kneighbors_graph
from torch_spline_conv import spline_conv
from torch import nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch.utils.data.sampler import SubsetRandomSampler
from lightning.pytorch.callbacks import StochasticWeightAveraging, LearningRateMonitor
import scipy.io
from utils.utils import create_folder_if_not_exists
from pytorch_lightning.loggers import TensorBoardLogger
dimD = 1119

batch_size =2
exp_num = '1'
device = 'cuda:0'
model_choise = 'nn'
ev = 'test'
dirpath = '/home/sv6234/ECGI/checkHybrid_testing'
create_folder_if_not_exists(dirpath)


def read_matrix(file_name):
    return np.fromfile(file_name)
def main():
    #mp.set_start_method('spawn') 
    #torch.set_float32_matmul_precision('medium')
    dimD = 1119
    par = torch.from_numpy(np.full((1862,1),0.15)).to(torch.float32).to(device)
    print(par.get_device())
    """a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
    S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float32).to(device)
    H=read_matrix('/home/sv6234/ECGI/data/EC/Trans.bin')
    H=torch.from_numpy(H.reshape(396,dimD)).to(torch.float32).to(device)"""
    mat = scipy.io.loadmat('S_Trans_new.mat')
    mat = mat['S']
    S=torch.from_numpy(mat).float().to(device)
    mat = scipy.io.loadmat('H_Trans_new.mat')
    mat = mat['H']
    H=torch.from_numpy(mat).float().to(device)
    
    dataset=TMPDatasetEQ("/home/sv6234/hyper-ep/data/TMP_data_UV_new/","/home/sv6234/hyper-ep/data/TMP_data_GT_new/",10,H=None)
    train_set, valid_set, test_set=torch.utils.data.random_split(dataset,[636,212,212])
    with open('/home/sv6234/hyper-ep/EC_642/Final/heart.idx', 'rb') as file:
        # Read data from the file using numpy.fromfile
        idx = np.fromfile(file, dtype=np.int32)
    ids=np.where(idx>0)
    ids=torch.tensor(ids).to(device)
    
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    #train_loader = torch.load('train_loader_checkHybrid_TMP_full.pt')
    #train_loader = torch.load('/home/sv6234/ECGI/train_loader_checkHybrid_EGM_in.pt')
    test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)
    #valid_loader = torch.load('valid_loader_checkHybrid_TMP_full.pt')
    valid_loader=DataLoader(valid_set,batch_size=batch_size,shuffle=False)
    #valid_loader = torch.load('/home/sv6234/ECGI/valid_loader_checkHybrid_EGM_in.pt')
    #test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)
    

    if model_choise=='nn':
        model = HybridModel(S, dimD, par, batch_size, ids,isPhysics=False,isRecon=False, surfaceTMP=False,outputSurface=False,adjoint=False,EGM_in=False,EGM_out=False,dilate_loss=False,use_resnet=True,seed=1120)
    elif model_choise=='only_nn':
        model = OnlyNN(S, dimD, par, batch_size)
    elif model_choise=='only_gcn':
        full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
        full_heart=full_heart.reshape((-1,3))
    
        a=kneighbors_graph(full_heart, 6)
        arr=a.toarray()

        points=np.where(arr==1)
        edges=np.reshape(points, (2,-1))
   

        dist = list(map(lambda x, y: np.linalg.norm(x-y), full_heart[edges[0]], full_heart[edges[1]]))
        normDist = dist/np.linalg.norm(dist) 
        


        #node_features=torch.rand((1862,32),requires_grad=True,dtype=torch.float32).to(device)
        #print(node_features.shape)
        edges=torch.from_numpy(edges).to(device)
    
        pseudo = torch.from_numpy(normDist).to(torch.float32).to(device)
    
    
        
        #weight = torch.rand((25,2,1),dtype=torch.float32,requires_grad=True).to(device).to(device)
        
    

         # only use open B-splines
        degree = 1  # B-spline degree of 1
        norm = True 
        model = OnlyGCN(S, dimD, par, batch_size,edges,pseudo,degree, norm,bias=None, root_weight=None)
        
    else:
        full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
        full_heart=full_heart.reshape((-1,3))
    
        a=kneighbors_graph(full_heart, 6)
        arr=a.toarray()

        points=np.where(arr==1)
        edges=np.reshape(points, (2,-1))
   

        dist = list(map(lambda x, y: np.linalg.norm(x-y), full_heart[edges[0]], full_heart[edges[1]]))
        normDist = dist/np.linalg.norm(dist) 
        


        #node_features=torch.rand((1862,32),requires_grad=True,dtype=torch.float32).to(device)
        #print(node_features.shape)
        edges=torch.from_numpy(edges).to(device)
    
        pseudo = torch.from_numpy(normDist).to(torch.float32).to(device)
    
    
        
        #weight = torch.rand((25,2,1),dtype=torch.float32,requires_grad=True).to(device).to(device)
        
    

         # only use open B-splines
        degree = 1  # B-spline degree of 1
        norm = True 
        model = HybridModel1(S, dimD, par, batch_size,edges,pseudo,degree, norm,bias=None, root_weight=None)
    #/home/sv6234/ECGI/checkHybrid_TMP_Full    
    #/home/sv6234/ECGI/checkHybrid_EGM_dilate_RES_lstm
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=dirpath,
    filename="best-model-{epoch:02d}-{val_loss:.3f}",
    mode="min",
    save_top_k=3)
    #params = list(model.ode.parameters())

# Normalize the parameters
    
    #StochasticWeightAveraging(swa_lrs=1e-2)
    #torch.save(train_loader, 'train_loader_checkHybrid_TMP_full.pt')
    #torch.save(valid_loader, 'valid_loader_checkHybrid_TMP_full.pt')
    #torch.save(test_loader, 'test_loader_checkHybrid_TMP_full.pt')
    #,StochasticWeightAveraging(swa_lrs=1e-2) 
    #gradient_clip_val=0.5 
    logger = TensorBoardLogger('lightning_logs', name='Full_TMP')         
    trainer = Trainer(logger=logger,default_root_dir='',limit_train_batches=int(636/batch_size),limit_val_batches=13,max_epochs=20,accelerator='gpu', devices=1,callbacks=[checkpoint_callback],
                        num_sanity_val_steps=0,gradient_clip_val=0.5)
    trainer.fit(model,train_loader, valid_loader)
    
   
if __name__=='__main__':
    main()
    