import os
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import torch
import lightning as L
from torch.nn import functional as F
import time
from torchdiffeq import  odeint
import numpy as np
#from torch_spline_conv import spline_conv
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from torch_geometric.nn.conv import spline_conv
from torch.nn import Linear, ReLU
from torch_geometric.nn import Sequential


device = 'cuda:0'

class APModel(nn.Module):
    def __init__(self, S, par, batch_size, dimD,nn) -> None:
        super(APModel, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.nn = nn
    def forward(self,t,y):
        with torch.enable_grad():
            u = torch.t(y[:,0:self.dimD])
                
            v = torch.t(y[:,self.dimD:self.dimD*2])
            
            k = 8
            e = 0.01
                #print(self.S,u)
            MK = torch.matmul(self.S, u)
                #print(torch.t(torch.cat((u,v))).shape)
            nn=self.nn(torch.t(torch.cat((u,v))))
                #print(nn.shape)
            pde1 = MK + k*u*(1-u)*(u-self.par) -torch.t(nn)
            pde2 = -e*(k*u*(u-self.par-1)+v)
                #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
                
        return torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))

class MetaNet(nn.Module):
    def __init__(self,):
        super(MetaNet, self).__init__()
        
        self.fc1= nn.Sequential(nn.Linear(3724, 64),nn.Dropout(0.1),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,128),nn.Dropout(0.1),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128,1862),nn.Dropout(0.1),nn.ReLU())
        
        
        
        
    def forward(self, x) -> Any:
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        
        
        return out
    
class ContextEncoder1(nn.Module):
    def __init__(self,):
        super(ContextEncoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1862, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3,1,1),
            nn.Flatten(),
            
            nn.Linear(3000 ,1),
            
            nn.Sigmoid()
        )
    
    def forward(self, x) -> Any:
        
        out = self.encoder(x)
        
        return out*(0.3) 
    
class ContextEncoder2(nn.Module):
    def __init__(self,):
        super(ContextEncoder2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1862, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3,1,1),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3,1,1),
            nn.Flatten(),
            
            nn.Linear(6000 ,1),
            
            nn.Sigmoid()
        )
    
    def forward(self, x) -> Any:
        
        out = self.encoder(x)
        
        return out*(0.3)      



class ContextEncoder(nn.Module):
    def __init__(self,):
        super(ContextEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1862, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(3,1,1),
            nn.Flatten(),
            
            nn.Linear(3000 ,8),
            nn.ReLU(),
            nn.Linear(8,1),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> Any:
        
        out = self.encoder(x)
        
        return out*(0.3)
    
class ContextEncoderLSTM(nn.Module):
    def __init__(self,input_dim=1862, seq_l=375,mid_dim=40, mid_1=20,out_dim=1, mid_d=64):
        super(ContextEncoderLSTM, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim)
        self.l2 = nn.LSTM(mid_dim,mid_1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ll1 = nn.Linear(mid_1*seq_l, mid_d)
        self.ll2 = nn.Linear(mid_d, out_dim)
        
    
    def forward(self, x) -> Any:
        batch, _, _ = x.size()
        out, h1 = self.l1(x)
        out = self.relu(out)
        out, h2 = self.l2(out)
        out = out.contiguous().view(batch,-1)
        out = self.ll1(out)
        out=self.relu(out)
        out = self.ll2(out)
        out = self.sigmoid(out)
        
        
        return out*0.5
        
        
class MetaNet1(nn.Module):
    def __init__(self,):
        super(MetaNet1, self).__init__()
        
        self.fc1= nn.Sequential(nn.Linear(1862, 64),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,128),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(128,1862))
        
        
        
    def forward(self, x) -> Any:
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out

    
class HybridModel(L.LightningModule):
    def __init__(self, S, dimD,par, batch_size):
        super().__init__()
        self.metaNet = MetaNet()
        self.S = S
        self.dimD = dimD
        self.par = par
        self.batch_size = batch_size
        self.ode = APModel(self.S,self.par, self.batch_size,self.dimD,self.metaNet)
        self.contextEnc = ContextEncoder1()
        t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
        val=0
        for i in range(375):
            t[i] = val
            val += 0.9
        self.t = t
        self.save_hyperparameters()
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(),lr=1e-3)
    
    def forward(self, x,k) -> Any:
        k =k.squeeze(1)
        k = k.permute(0,2,1)
        
        par = self.contextEnc(k)
        #print(par)
        #par=par.squeeze()
        p=torch.zeros((1862,x.shape[0]))
        #print(par)
        
        for i in range(x.shape[0]):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()
        self.par = p.to(device)   
        t0 = time.time()
        self.ode.par = self.par
        TMP=odeint(self.ode, x, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        return TMP
    
    
    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        
        UV, y, k, a = train_batch
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        #k =k.squeeze(1)
        #k = k.permute(0,2,1)
        #print("time for context")
        
        bs, num_k, seq_len, dim=k.size()
        par=torch.zeros(bs,1).to(device)
        for i in range(num_k):
            temp=k[:,i,:,:]
            print(temp.shape)
            par += self.contextEnc(temp.permute(0,2,1))
        par = par/num_k    
        t0 = time.time()
        #par = self.contextEnc(k)
        #print("--- %s seconds ---" % (time.time() - t0))
        #print(par)
        #par=par.squeeze()
        p=torch.zeros((1862,self.batch_size))
        #print(par)
        t0 = time.time()
        for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()
        #print("--- %s seconds ---" % (time.time() - t0))
        self.par = p.to(device)   
        #print("time for ode")
        t0 = time.time()
        self.ode.par = self.par
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        #print("--- %s seconds ---" % (time.time() - t0))
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        
        param_loss = F.mse_loss(a, par.squeeze())
        loss = F.mse_loss(TMP, y)
        self.log_dict({"train": loss, "par": param_loss},prog_bar=True)
        
        
        
        return loss
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        UV, y, k, a = train_batch
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        k =k.squeeze(1)
        k = k.permute(0,2,1)
        par = self.contextEnc(k)
        #print(par)
        #par=par.squeeze()
        p=torch.zeros((1862,self.batch_size))
        #print(par)
        
        for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()
        self.par = p.to(device)   
        t0 = time.time()
        self.ode = APModel(self.S,self.par,self.batch_size,self.dimD,self.metaNet)
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        
        param_loss = F.mse_loss(a, par.squeeze())
        loss = F.mse_loss(TMP, y)
        self.log_dict({"val_loss": loss, "par": param_loss},prog_bar=True)
        
        return loss
    
    def test_step(self,train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        UV, y, k, a = train_batch
        
        t0 = time.time()
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        
        
        loss = F.mse_loss(TMP, y)
        
        
        
        self.log('test', loss, prog_bar=True)
        return loss
   
    

    

class APModel1(nn.Module):
    def __init__(self, S, par, batch_size, dimD,edges,pseudo,degree,norm,bias,root_weight) -> None:
        super(APModel1, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.edges =edges
        self.pseudo = pseudo
        
        self.degree = degree
        self.norm = norm
        self.bias = bias
        self.root_weight = root_weight
        #self.fcn = MetaNet1()
        """self.nn = Sequential('x, edge_index, pseudo', [
    (spline_conv.SplineConv(2,2,1,3), 'x, edge_index, pseudo -> x'),
    ReLU(inplace=True),
    (spline_conv.SplineConv(2,1,1,3), 'x, edge_index, pseudo -> x'),
    #ReLU(inplace=True),
    #Linear(64, out_channels),
    ])"""
        self.nn = spline_conv.SplineConv(2,1,1,7)
        #self.nn1 = spline_conv.SplineConv(2,1,1,3)
        self.expaned_edges, self.expanded_pseudo = self.expand(batch_size, edges,pseudo)
        
    def expand(self, batch_size, edges, pseudo):
        n=self.dimD
        expanded_edges=edges
        expanded_pseudo = pseudo
        for i in range(batch_size-1):
            print(torch.add(edges,n))
            expanded_edges=torch.cat((expanded_edges,torch.add(edges,n)),dim=1)
            expanded_pseudo=torch.cat((expanded_pseudo,pseudo),dim=0)
            n=n+self.dimD
        return expanded_edges, expanded_pseudo
            
    """def expand(self, batch_size, num_nodes, edge_index, edge_attr, sample_rate=None):
    # edge_attr = edge_attr.repeat(T, 1)
        num_edges = int(edge_index.shape[1] / batch_size)
        edge_index = edge_index[:, 0:num_edges]
        edge_attr = edge_attr[0:num_edges, :]


        sample_number = int(sample_rate * num_edges) if sample_rate is not None else num_edges
        selected_edges = torch.zeros(edge_index.shape[0], batch_size * sample_number).to('cuda:0')
        selected_attrs = torch.zeros(batch_size  * sample_number, edge_attr.shape[1]).to('cuda:0')

        for i in range(batch_size ):
            chunk = edge_index + num_nodes * i
            if sample_rate is not None:
                index = np.random.choice(num_edges, sample_number, replace=False)
                index = np.sort(index)
            else:
                index = np.arange(num_edges)
            print(chunk.shape)
            selected_edges[:, sample_number * i:sample_number * (i + 1)] = chunk[:, index]
            selected_attrs[sample_number * i:sample_number * (i + 1), :] = edge_attr[index, :]

        selected_edges = selected_edges
        return selected_edges, selected_attrs"""
    
    def forward(self,t,y):
        with torch.enable_grad():
            #print(self.expaned_edges.shape, self.expanded_pseudo)
            u = torch.t(y[:,0:self.dimD])
            
            v = torch.t(y[:,self.dimD:self.dimD*2])
            
            k = 8
            e = 0.01
            #print(self.S,u)
            MK = torch.matmul(self.S, u)
            u1 = torch.flatten(u).unsqueeze(0)
            v1 = torch.flatten(v).unsqueeze(0)
            uv = torch.cat((u1,v1),0)
            #out = self.nn(torch.cat((u,v),1),self.edges,torch.t(self.pseudo.unsqueeze(0)))
            
            """for i in range(self.batch_size):
                u1 = u[:,i].unsqueeze(0)
                v1 = v[:,i].unsqueeze(0)
                #print(torch.cat((u1,v1),0).shape)
                #out= torch.nn.functional.normalize(out)
                if i==0:
                    uv=torch.cat((u1,v1),0)
                else:
                    uv = torch.cat((uv, torch.cat((u1,v1),0)),1)"""
                
                    
                    
            
            out = self.nn(torch.t(uv),self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
            #out = self.nn1(out, self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
            
            
            pde1 = MK + k*u*(1-u)*(u-self.par) + out.view(1862,-1)
            pde2 = -e*(k*u*(u-self.par-1)+v)
            #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
            return torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))
class GCN(nn.Module):
    def __init__(self, edges,pseudo,degree,norm,bias = None, root_weight=None) -> None:
        super().__init__()
        self.edges = edges
        self.weight = nn.Parameter(torch.Tensor(25, 2, 1),requires_grad=True).to(device)
        self.kernel_size = torch.tensor([25]).to('cuda:0') 
        self.is_open_spline = torch.tensor([1], dtype=torch.uint8).to(device)
        self.pseudo = pseudo
        self.degree = degree
        self.norm = norm
        self.bias = self.bias = nn.Parameter(torch.Tensor(1))
        self.root_weight = nn.Parameter(torch.Tensor(2,1))
    def forward(self,x):
        out = spline_conv(x,self.edges,self.pseudo,self.weight,self.kernel_size,self.is_open_spline,self.degree,self.norm)
        return out
        
    
    
class HybridModel1(L.LightningModule):
    def __init__(self, S, dimD,par, batch_size,edges,pseudo,degree, norm,bias=None, root_weight=None):
        super().__init__()
        
        self.S = S
        self.dimD = dimD
        self.par = par
        self.batch_size = batch_size
        
        t=torch.from_numpy(np.zeros(200)).to(torch.float32).to(device)
        val=0
        for i in range(200):
            t[i] = val
            val += 0.9
        self.t = t
        
        self.ode = APModel1(self.S,self.par, self.batch_size,self.dimD,edges,pseudo,degree,norm,bias,root_weight)
        self.save_hyperparameters()
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(),lr=1e-3)
    
    def forward(self, x) -> Any:
        TMP=odeint(self.ode, x, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        return TMP
    
    
    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        UV, y = train_batch
        
        t0 = time.time()
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        
        
        loss = F.mse_loss(TMP, y)
        
        self.log('train', loss,prog_bar=True)
        
        
        return loss
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        UV, y = train_batch
        
        t0 = time.time()
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
       
        
        loss = F.mse_loss(TMP, y)
        
        
        
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def test_step(self,train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        UV, y = train_batch
        
        t0 = time.time()
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:, :, 0:self.dimD].permute(1,0,2)
        
        
        loss = F.mse_loss(TMP, y)
        
        
        
        self.log('test', loss, prog_bar=True)
        return loss
        
            



        
            
"""    def APModel(self,y, S, dimD, par):
        
        
        u = y[0:dimD,:]
        
        
        v = y[dimD:dimD*2, :]
        
        
        k = 8
        e = 0.01
        MK = torch.matmul(S, u)
        nn=self.metaNet(torch.t(torch.cat((u,v),1)))
        
        pde1 = MK + k*u*(1-u)*(u-par) + nn.reshape(1862,self.batch_size)
        pde2 = -e*(k*u*(u-par-1)+v)
        
        return torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0)


    def RK5(self, y, h, fun, S, dimD, par):
        y = torch.t(y)
        k1 = fun(y, S, dimD, par)
        k2 = fun(y + h*k1/4, S, dimD, par)
        k3 = fun(y + h*k1*3/32 + h*k2*9/32, S, dimD, par)
        k4 = fun(y + h*k1*1932/2197 + h*k2*-7200/2197 + h*k3*7296/2197, S, dimD, par)
        k5 = fun(y + h*k1*439/216 + h*k2*-8 + h*k3*3680/513 + h*k4*-845/4104, S, dimD, par)
        k6 = fun(y + h*k1*-8/27 + h*k2*2 + h*k3*-3544/2565 + h*k4*1859/4104 + h*k5*-11/40, S, dimD, par)
        dy = h * (k1*16/135 + k2*0 + k3*6656/12825 + k4*28561/56430 + k5*-9/50 + k6*2/55)
        return dy"""