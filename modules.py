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
from torchdiffeq import odeint
import numpy as np
#from torch_spline_conv import spline_conv
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from torch_geometric.nn.conv import spline_conv
from torch.nn import Linear, ReLU,Tanh
from torch_geometric.nn import Sequential
from lightning.pytorch.utilities import grad_norm
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#import pytorch_warmup as warmup
from torch.optim.lr_scheduler import ExponentialLR
from scipy.integrate import solve_ivp, LSODA
import torchode as to
from TorchDiffEqPack import  odesolve_adjoint
#from ignite.handlers import create_lr_scheduler_with_warmup
from torchdyn.core import NeuralODE
from torch_geometric.nn.conv import GCNConv
import torch.nn.init as init


device = 'cuda:0'

class APModelPhysics(nn.Module):
    def __init__(self, S, par, batch_size, dimD, nn) -> None:
        super(APModelPhysics, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.nn = nn
        #self.context = context
        #self.p = p
        
        
        
        
    def forward(self,t,y):
        #print(self.nn.parameters())
        #with torch.enable_grad():
        
            
        u = torch.t(y[:,0:self.dimD])
                        
        v = torch.t(y[:,self.dimD:self.dimD*2])
                    
        k = 8
        e = 0.01
                        #print(self.S,u)
        MK = torch.matmul(self.S, u)
        
        #nn=self.nn(torch.t(torch.cat((u,v))))
        
                            #print(nn.shape)
            #nn=torch.nn.Parameter(nn)
        #change this back    
        pde1 = MK + k*u*(1-u)*(u-self.par) #-torch.t(nn)
        pde2 = -e*(k*u*(u-self.par-1)+v)
                        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        val=torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))
         
        return val
    
class APModel_uv(nn.Module):
    def __init__(self, S, par, batch_size, dimD, nn,adjoint=False) -> None:
        super(APModel_uv, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.nn = nn
        self.retained_vars = {}
        self.adjoint = adjoint
        #self.context = context
        #self.p = p
        
        
        
        
    def forward(self,t,y):
        #print(self.nn.parameters())
        #with torch.enable_grad():
        
        #nn = self.nn(y[:,:self.dimD*2])
        uv = torch.cat((y[:,0:self.dimD].unsqueeze(2),y[:,self.dimD:self.dimD*2].unsqueeze(2)),2).view(-1,2) 
        u = torch.t(y[:,0:self.dimD])
        v = torch.t(y[:,self.dimD:self.dimD*2])
        
         #bs,1862,2
           
        nn=self.nn(uv).view(self.batch_size,self.dimD,1).squeeze()   
        k = 8
        e = 0.01
                        #print(self.S,u)
        MK = torch.matmul(self.S, u)
        
        #nn=self.nn(torch.t(torch.cat((u,v))))
        
                            #print(nn.shape)
            #nn=torch.nn.Parameter(nn)
        
        if self.adjoint:
            par = y[:,self.dimD*2:].squeeze() 
            pde1 = MK + k*u*(1-u)*(u-par) -torch.t(nn)
            pde2 = -e*(k*u*(u-par-1)+v)
                        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        #p=self.par.clone()
            p = torch.zeros((self.batch_size,1)).to(device)
        else:
            pde1 = MK + k*u*(1-u)*(u-self.par) -torch.t(nn)
            pde2 = -e*(k*u*(u-self.par-1)+v)
        val=torch.t(torch.cat((pde1, pde2), dim=0))
        if self.adjoint:
            val = torch.cat((val, p), dim = 1)
         
        return val
    

class APModel(nn.Module):
    def __init__(self, S, par, batch_size, dimD, nn,adjoint=False) -> None:
        super(APModel, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.nn = nn
        self.retained_vars = {}
        self.adjoint = adjoint
        #self.context = context
        #self.p = p
        self.t1 =0
        
        
        
        
    def forward(self,t,y):
        #print(self.nn.parameters())
        #with torch.enable_grad():
        self.t1+=1
        nn = self.nn(y[:,:self.dimD*2])
            
        u = torch.t(y[:,0:self.dimD])
        
                        
        v = torch.t(y[:,self.dimD:self.dimD*2])
        
                    
        k = 8
        e = 0.01
                        #print(self.S,u)
        MK = torch.matmul(self.S, u)
        
        #nn=self.nn(torch.t(torch.cat((u,v))))
        
                            #print(nn.shape)
            #nn=torch.nn.Parameter(nn)
        if self.adjoint:
            par = y[:,self.dimD*2:].squeeze() 
            pde1 = MK + k*u*(1-u)*(u-par) -torch.t(nn)
            pde2 = -e*(k*u*(u-par-1)+v)
                        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        #p=self.par.clone()
            p = torch.zeros((self.batch_size,1)).to(device)
        else:
            pde1 = MK + k*u*(1-u)*(u-self.par) -torch.t(nn)
            pde2 = -e*(k*u*(u-self.par-1)+v)
        val=torch.t(torch.cat((pde1, pde2), dim=0))
        if self.adjoint:
            val = torch.cat((val, p), dim = 1)
         
        return val
    

#3724,64,128,1862 # activation
class MetaNet(nn.Module):
    def __init__(self):
        super(MetaNet, self).__init__()
        
        self.fc1= nn.Sequential(nn.Linear(3724, 64) ,nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,128),nn.ReLU())
        #self.fc3 = nn.Sequential(nn.Linear(64,128),nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(128,1862),nn.Tanh())
        
        
        
    def forward(self, x) -> Any:
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        #out = self.fc4(out)
        
        return out
    
class MetaNet_new(nn.Module):
    def __init__(self):
        super(MetaNet_new, self).__init__()
        
        self.fc1= nn.Sequential(nn.Linear(2, 16) ,nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(16,64),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64,16),nn.SiLU())
        self.fc4 = nn.Sequential(nn.Linear(16,1),nn.Tanh())
        
        
        
    def forward(self, x) -> Any:
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        
        return out
    
class MetaNet_new_v1(nn.Module):
    def __init__(self):
        super(MetaNet_new_v1, self).__init__()
        self.net = nn.Sequential(nn.Linear(2, 16) ,nn.ReLU(),
                                nn.Linear(16,64),nn.ReLU(),
                                nn.Linear(64,128),nn.ReLU(),
                                nn.Linear(128,64),nn.ReLU(),
                                nn.Linear(64,16),nn.ReLU(),
                                nn.Linear(16,1),nn.Tanh())
        
        
        
        
    def forward(self, x) -> Any:
        
        out = self.net(x)
        
        return out
    
import torch.nn as nn

class MetaNet_new_v2(nn.Module):
    def __init__(self):
        super(MetaNet_new_v2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            #nn.Dropout(p=0.1),  # Dropout after the first ReLU
            
            nn.Linear(16, 64),
            nn.ReLU(),
            #nn.Dropout(p=0.2),  # Dropout after the second ReLU
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout after the third ReLU
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),  # Dropout after the fourth ReLU
            
            nn.Linear(64, 16),
            nn.ReLU(),
            #nn.Dropout(p=0.1),  # Dropout after the fifth ReLU
            
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
    def forward(self, x) -> Any:
        out = self.net(x)
        return out

    
class ContextEncoderLSTM(nn.Module):             #40     #128   #32 #20
    def __init__(self,input_dim=1862, seq_l=375,mid_dim=128, mid_1=32,out_dim=1, mid_d=64):
        super(ContextEncoderLSTM, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim,batch_first=True)
        self.l2 = nn.LSTM(mid_dim,mid_1,batch_first=True)
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
        out = (self.sigmoid(out)*(0.16-0.08))+ 0.08
        
        
        
        return out
    
    
class ContextEncoderLSTMSurface(nn.Module):             #40         #20
    def __init__(self,input_dim=477, seq_l=375,mid_dim=128, mid_1=32,out_dim=1, mid_d=64):
        super(ContextEncoderLSTMSurface, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim,batch_first=True)
        self.l2 = nn.LSTM(mid_dim,mid_1,batch_first=True)
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
        out = (self.sigmoid(out)*(0.16-0.08))+ 0.08
        
        
        
        return out
    
    
class MultiLayerLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=396, hidden_dim=128, mid_dim = 64,output_dim=1, num_layers=3):
        super(MultiLayerLSTMWithAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid() 
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.fc_1 = nn.Linear(hidden_dim, mid_dim)
        self.fc_2 = nn.Linear(mid_dim, output_dim)
    
    def forward(self, x):
        #h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x)
        
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        
        out = self.fc_1(context)
        out = self.relu(out)
        out = self.fc_2(out)
        out = (self.sigmoid(out)*(0.16-0.08))+ 0.08
        return out    
class ContextEncoderLSTM_EGM(nn.Module):             #128 change        #20
    def __init__(self,input_dim=396, seq_l=375,mid_dim=128, mid_1=32,out_dim=1, mid_d=64):
        super(ContextEncoderLSTM_EGM, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim,batch_first=True)
        self.l2 = nn.LSTM(mid_dim,mid_1,batch_first=True)
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
        out = (self.sigmoid(out)*(0.16-0.08))+ 0.08
        
        
        
        return out
    
class ContextEncoderLSTM_EGM_out(nn.Module):             #128 change        #20
    def __init__(self,input_dim=396, seq_l=375,mid_dim=128, mid_1=32,out_dim=1, mid_d=64):
        super(ContextEncoderLSTM_EGM_out, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim,batch_first=True)
        self.l2 = nn.LSTM(mid_dim,mid_1,batch_first=True)
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
        out = (self.sigmoid(out)*(0.16-0.08))+ 0.08
        
        
        
        return out
    
class ContextEncoderLSTM_EGM_out_1(nn.Module):             #128 change        #20
    def __init__(self,input_dim=396, seq_l=375,mid_dim_2=128,mid_dim_3=64,mid_dim_1=512, mid_1=32,out_dim=1, mid_d=64):
        super(ContextEncoderLSTM_EGM_out_1, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim_1,batch_first=True)
        self.l2 = nn.LSTM(mid_dim_1,mid_dim_2,batch_first=True)
        self.l3 = nn.LSTM(mid_dim_2,mid_dim_3,batch_first=True)
        self.l4 = nn.LSTM(mid_dim_3,mid_1,batch_first=True)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ll1 = nn.Linear(mid_1*seq_l, mid_d)
        self.ll2 = nn.Linear(mid_d, out_dim)
        
    
    def forward(self, x) -> Any:
        batch, _, _ = x.size()
        
        out, h1 = self.l1(x)
        
        out = self.relu(out)
        out, h2 = self.l2(out)
        out = self.relu(out)
        out , h3 = self.l3(out)
        out = self.relu(out)
        out, h4 = self.l4(out)
        out = out.contiguous().view(batch,-1)
        out = self.ll1(out)
        out=self.relu(out)
        out = self.ll2(out)
        out = (self.sigmoid(out)*(0.16-0.08))+ 0.08
        return out
    
 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_channels=396, num_classes=128):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        self.sigmoid = nn.Sigmoid()
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = (self.sigmoid(x)*(0.16-0.08))+ 0.08
        return x
    
class ContextEncoderLSTM1(nn.Module):
    def __init__(self,input_dim=1862, seq_l=375,mid_dim=128,mid_dim2=20, mid_1=64,out_dim=1, mid_d=64):
        super(ContextEncoderLSTM1, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim,batch_first=True)
        self.l2 = nn.LSTM(mid_dim,mid_1,batch_first=True)
        self.l3 = nn.LSTM(mid_1,mid_dim2,batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ll1 = nn.Linear(mid_dim2*seq_l, mid_d)
        self.ll2 = nn.Linear(mid_d, out_dim)
        
    
    def forward(self, x) -> Any:
        batch, _, _ = x.size()
        
        out, h1 = self.l1(x)
        
        out = self.relu(out)
        out, h2 = self.l2(out)
        out = self.relu(out)
        out, h3 = self.l3(out)
        out = self.relu(out)
        out = out.contiguous().view(batch,-1)
        out = self.ll1(out)
        out=self.relu(out)
        out = self.ll2(out)
        out = self.sigmoid(out)*(0.16-0.08)+ 0.08
        
        
        
        return out
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, middle, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, middle)
        self.conv2 = GCNConv(middle, out_channels)

    def forward(self, features, edge):
        x, edge_index = features, edge
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x
    

class APModel1(nn.Module,):
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
        self.t =0
        #self.fcn = MetaNet1()
        
        """self.nn = Sequential('x, edge_index, pseudo', [
    (spline_conv.SplineConv(2,16,1,6), 'x, edge_index, pseudo -> x'),
    ReLU(),
    (spline_conv.SplineConv(16,1,1,6), 'x, edge_index, pseudo -> x'),
   
    Tanh()
    
    #Linear(64, out_channels),
    ])"""
        self.nn = spline_conv.SplineConv(2,1,1,5)
        
        #self.nn = GCN(2, 1, 1)
        self.act = nn.Tanh()
        #self.nn1 = spline_conv.SplineConv(2,1,1,3)
        self.expaned_edges, self.expanded_pseudo = self.expand(batch_size, edges,pseudo)
        #self.initialize_weights()
    def initialize_weights(self):
        for layer in self.nn:
            if isinstance(layer, spline_conv.SplineConv):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        
    def expand(self, batch_size, edges, pseudo):
        n=self.dimD
        expanded_edges=edges
        expanded_pseudo = pseudo
        for i in range(batch_size-1):
            print(torch.add(edges,n))
            expanded_edges=torch.cat((expanded_edges,torch.add(edges,n)),dim=1)
            expanded_pseudo=torch.cat((expanded_pseudo,pseudo),dim=0)
            n=n+self.dimD
        return expanded_edges.to(device), expanded_pseudo.to(device)
            
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
        self.t += 1
            #print(self.expaned_edges.shape, self.expanded_pseudo)
        u = torch.t(y[:,0:self.dimD]) #1862 , bs
            
        v = torch.t(y[:,self.dimD:self.dimD*2])
        par = y[:,self.dimD*2:].squeeze()    
        k = 8
        e = 0.01
            #print(self.S,u)
        MK = torch.matmul(self.S, u)
        u1 = torch.flatten(torch.t(u)).unsqueeze(0) #1,bs*1862
        v1 = torch.flatten(torch.t(v)).unsqueeze(0)
        uv = torch.cat((u1,v1),0)#2,bs*1862
            #out = self.nn(torch.cat((u,v),1),self.edges,torch.t(self.pseudo.unsqueeze(0)))
            
        """for i in range(self.batch_size):
                u1 = u[:,i].unsqueeze(0)
                v1 = v[:,i].unsqueeze(0)
                #print(torch.cat((u1,v1),0).shape)
                #out= torch.nn.functional.normalize(out)
                if i==0:
                    uv1=torch.cat((u1,v1),0)
                else:
                    uv1= torch.cat((uv1, torch.cat((u1,v1),0)),1)
        print(torch.equal(uv,uv1))"""
                
                    
                    
        #print(self.expaned_edges.shape)    
        
        out = self.nn(torch.t(uv),self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
        out=torch.unflatten(out,dim=0,sizes=(self.dimD,self.batch_size)).squeeze()
        #out = self.nn(torch.t(uv),self.expaned_edges)
            #out = self.nn1(out, self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
        #out = self.act(out)
        #out = torch.nn.Parameter(out)
            
        pde1 = MK + k*u*(1-u)*(u-par) - out
        pde2 = -e*(k*u*(u-par-1)+v)
        
        val=torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))
        p = torch.zeros((self.batch_size,1)).to(device)
        val = torch.cat((val,p),dim=1)
            #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        return val
    
class APModel2(nn.Module):
    def __init__(self, S, par, batch_size, dimD, nn) -> None:
        super(APModel2, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.nn = nn
        #self.context = context
        #self.p = p
        
        
        
        
    def forward(self,t,y,args=None):
        #print(self.nn.parameters())
        #with torch.enable_grad():
        
            
        u = torch.t(y[:,0:self.dimD])
                        
        
                    
        
        nn=self.nn(torch.t(u))
        
                            #print(nn.shape)
            #nn=torch.nn.Parameter(nn)
          
        #pde1=torch.t(nn)
        
                        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        val=nn.squeeze()
         
        return val
    

    
    
class MetaNet4(nn.Module):
    def __init__(self):
        super(MetaNet4, self).__init__()
        self.fc1= nn.Sequential(nn.Linear(1862, 512) ,nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(512,256),nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(256,128),nn.SiLU())
        self.fc4 = nn.Sequential(nn.Linear(128,256),nn.SiLU())
        self.fc5 = nn.Sequential(nn.Linear(256,512),nn.SiLU())
        self.fc6 = nn.Sequential(nn.Linear(512,1862),nn.SiLU())
        
        
        
    def forward(self, x) -> Any:
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        #out = self.fc4(out)
        
        return out
        
        
    
class MetaNet2(nn.Module):
    def __init__(self,bs):
        self.bs = bs
        super(MetaNet2, self).__init__()
        self.meta_net = nn.Sequential(nn.Linear(1,32),nn.LeakyReLU(),
                                      nn.Linear(32,32),nn.LeakyReLU(),
                                      nn.Linear(32,64),nn.LeakyReLU(),
                                      nn.Linear(64,64),nn.LeakyReLU(),
                                      nn.Linear(64,2237894),nn.LeakyReLU())#2234368
        #self.fc1= nn.Sequential(nn.Linear(3724, 64) ,nn.ReLU())
        #self.fc2 = nn.Sequential(nn.Linear(64,128),nn.ReLU())
        #self.fc3 = nn.Sequential(nn.Linear(64,128),nn.SiLU())
        #self.fc3 = nn.Sequential(nn.Linear(128,1862),nn.Tanh())
        
    def set_param(self,param):
        param=torch.unsqueeze(param,dim=0).view(self.bs,1)
        weights = self.meta_net(param)
        self.net_weights, self.net_bias = weights[:,:2234368], weights[:,2234368:]
        
    def forward(self, x) -> Any:
        x=torch.unsqueeze(x,dim=1)
        state = x.view(self.bs,1,-1)
        #state = torch.unsqueeze(x,dim=0).view(self.bs, 1, -1)
        x = torch.bmm(state, self.net_weights[:,:1862*512].view(-1,1862,512))
        x = x + + self.net_bias[:, 0:512].view(-1,1,512)
        x = torch.nn.functional.silu(x)
        
        x = torch.bmm(x, self.net_weights[:,1862*512:1862*512+512*256].view(-1,512,256))
        x = x+ self.net_bias[:,512:512+256].view(-1,1,256)
        x = torch.nn.functional.silu(x)
        
        x = torch.bmm(x, self.net_weights[:,1862*512+512*256:1862*512+512*256+256*128].view(-1,256,128))
        x = x+ self.net_bias[:,512+256:512+256+128].view(-1,1,128)
        x = torch.nn.functional.silu(x)
        
        x = torch.bmm(x, self.net_weights[:,1862*512+512*256+256*128:1862*512+512*256+256*128+128*256].view(-1,128,256))
        x = x+ self.net_bias[:,512+256+128:512+256+128+256].view(-1,1,256)
        x = torch.nn.functional.silu(x)
        
        x = torch.bmm(x, self.net_weights[:,1862*512+512*256+256*128+128*256:1862*512+512*256+256*128+128*256+256*512].view(-1,256,512))
        x = x+ self.net_bias[:,512+256+128+256:512+256+128+256+512].view(-1,1,512)
        x = torch.nn.functional.silu(x)
        
        x = torch.bmm(x, self.net_weights[:,1862*512+512*256+256*128+128*256+256*512:1862*512+512*256+256*128+128*256+256*512+512*1862].view(-1,512,1862))
        x = x+ self.net_bias[:,512+256+128+256+512:512+256+128+256+512+1862].view(-1,1,1862)
        x = torch.nn.functional.tanh(x)
        return x
class APModel2(nn.Module,):
    def __init__(self, S, par, batch_size, dimD,edges,pseudo,degree,norm,bias,root_weight) -> None:
        super(APModel2, self).__init__()
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
        self.t = 0
        self.nn = Sequential('x, edge_index, pseudo', [
    (spline_conv.SplineConv(1,16,1,6), 'x, edge_index, pseudo -> x'),
    ReLU(),
    (spline_conv.SplineConv(16,32,1,6), 'x, edge_index, pseudo -> x'),
    ReLU(),
    (spline_conv.SplineConv(32,16,1,6), 'x, edge_index, pseudo -> x'),
    ReLU(),
    (spline_conv.SplineConv(16,1,1,6), 'x, edge_index, pseudo -> x'),
    Tanh()
    
    #Linear(64, out_channels),
    ])
        #self.nn = spline_conv.SplineConv(1,1,1,7)
        
        #self.nn = GCN(2, 1, 1)
        self.act = nn.Tanh()
        #self.nn1 = spline_conv.SplineConv(2,1,1,3)
        self.expaned_edges, self.expanded_pseudo = self.expand(batch_size, edges,pseudo)
        #self.initialize_weights()
    def initialize_weights(self):
        for layer in self.nn:
            if isinstance(layer, spline_conv.SplineConv):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)
        
    def expand(self, batch_size, edges, pseudo):
        n=self.dimD
        expanded_edges=edges
        expanded_pseudo = pseudo
        for i in range(batch_size-1):
            print(torch.add(edges,n))
            expanded_edges=torch.cat((expanded_edges,torch.add(edges,n)),dim=1)
            expanded_pseudo=torch.cat((expanded_pseudo,pseudo),dim=0)
            n=n+self.dimD
        return expanded_edges.to(device), expanded_pseudo.to(device)
            
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
        self.t += 1
            #print(self.expaned_edges.shape, self.expanded_pseudo)
        u = torch.t(y[:,0:self.dimD]) #1862 , bs
            
        
        
        k = 8
        e = 0.01
            #print(self.S,u)
        #MK = torch.matmul(self.S, u)
        u1 = torch.flatten(torch.t(u)).unsqueeze(0) #1,bs*1862
        
        
            #out = self.nn(torch.cat((u,v),1),self.edges,torch.t(self.pseudo.unsqueeze(0)))
            
        """for i in range(self.batch_size):
                u1 = u[:,i].unsqueeze(0)
                v1 = v[:,i].unsqueeze(0)
                #print(torch.cat((u1,v1),0).shape)
                #out= torch.nn.functional.normalize(out)
                if i==0:
                    uv1=torch.cat((u1,v1),0)
                else:
                    uv1= torch.cat((uv1, torch.cat((u1,v1),0)),1)
        print(torch.equal(uv,uv1))"""
                
                    
                    
        #print(self.expaned_edges.shape)    
        
        out = self.nn(torch.t(u1),self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
        out=torch.unflatten(out,dim=0,sizes=(self.dimD,self.batch_size)).squeeze()
        #out = self.nn(torch.t(uv),self.expaned_edges)
            #out = self.nn1(out, self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
        #out = self.act(out)
        #out = torch.nn.Parameter(out)
            
       
            #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        return torch.t(out)
    
        