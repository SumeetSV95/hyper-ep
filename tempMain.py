from model import HybridModel, HybridModel1
from lightning.pytorch import Trainer
from dataset import TMPDataset
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

dimD = 1862

batch_size =16

device = 'cuda:0'
model_choise = 'nn'


def read_matrix(file_name):
    return np.fromfile(file_name)
def main():
    #mp.set_start_method('spawn') 
    #torch.set_float32_matmul_precision('medium')
    dimD = 1862
    par = torch.from_numpy(np.full((1862,1),0.15)).to(torch.float32).to(device)
    print(par.get_device())
    a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
    S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float32).to(device)
    dataset=TMPDataset("/home/sv6234/ECGI/data/TMP_data_UV_new/","/home/sv6234/ECGI/data/TMP_data_GT_new/",1)
    train_set, valid_set, test_set=torch.utils.data.random_split(dataset,[1056,352,352])
    
    train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)
    
    #test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=False)
    valid_loader=DataLoader(valid_set,batch_size=batch_size,shuffle=False)

    if model_choise=='nn':
        model = HybridModel(S, dimD, par, batch_size)
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
    checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="checkpoints",
    filename="best-model-{epoch:02d}-{val_loss:.2f}",
    mode="min",
    save_top_k=3)
    
    trainer = Trainer(default_root_dir='',limit_train_batches=int(1056/batch_size),limit_val_batches=18,max_epochs=100,accelerator='gpu', devices=1,callbacks=[checkpoint_callback],
                        num_sanity_val_steps=0,accumulate_grad_batches=1)
    trainer.fit(model,train_loader, valid_loader)
   
if __name__=='__main__':
    main()