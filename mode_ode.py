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
from torch.nn import Linear, ReLU
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
class APModel(nn.Module):
    def __init__(self, S, par, batch_size, dimD, nn) -> None:
        super(APModel, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
        self.nn = nn
        self.retained_vars = {}
        #self.context = context
        #self.p = p
        
        
        
        
    def forward(self,t,y):
        #print(self.nn.parameters())
        #with torch.enable_grad():
        if isinstance(t, torch.Tensor) and t.requires_grad:
            self.retained_vars[t.item()] = self.par.clone().detach().requires_grad_(True)
        nn = self.nn(y[:,:self.dimD*2])
            
        u = torch.t(y[:,0:self.dimD])
        par = y[:,self.dimD*2:].squeeze()
                        
        v = torch.t(y[:,self.dimD:self.dimD*2])
                    
        k = 8
        e = 0.01
                        #print(self.S,u)
        MK = torch.matmul(self.S, u)
        
        #nn=self.nn(torch.t(torch.cat((u,v))))
        
                            #print(nn.shape)
            #nn=torch.nn.Parameter(nn)
          
        pde1 = MK + k*u*(1-u)*(u-par) -torch.t(nn)
        pde2 = -e*(k*u*(u-par-1)+v)
                        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        p=self.par.clone()
        p = p.unsqueeze(1)
        val=torch.t(torch.cat((pde1, pde2), dim=0))
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
    
class MetaNet6(nn.Module):
    def __init__(self):
        super(MetaNet6, self).__init__()
        self.fc1= nn.Sequential(nn.Linear(3724, 512) ,nn.SiLU())
        self.fc2 = nn.Sequential(nn.Linear(512,256),nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(256,128),nn.SiLU())
        self.fc4 = nn.Sequential(nn.Linear(128,256),nn.SiLU())
        self.fc5 = nn.Sequential(nn.Linear(256,512),nn.SiLU())
        self.fc6 = nn.Sequential(nn.Linear(512,1862),nn.Tanh())
        
        
        
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
    def __init__(self, *args, **kwargs) -> None:
        super(MetaNet2, self).__init__()
        self.nn = nn.Sequential(nn.Linear(3724,128),nn.Tanh(),
                                nn.Linear(128,64), nn.Tanh(),
                                nn.Linear(64,128), nn.Tanh(),
                                nn.Linear(128, 1862), nn.Tanh())
    def forward(self,x):
        x=torch.nn.functional.normalize(x)
        return self.nn(x)
           
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
        
        


    
class HybridModel(L.LightningModule):
    def __init__(self, S, dimD,par, batch_size, sufaceIds=None,isPhysics = False, isNeural=False, isRecon= False,surfaceTMP=False,outputSurface = False):
        super().__init__()
        self.metaNet = MetaNet()
        self.metaNet.apply(self.init_weights)
        self.S = S
        self.dimD = dimD
        self.par = par
        self.batch_size = batch_size
        self.isRecon = isRecon
        self.surfaceIds = sufaceIds
        self.isSurfaceTMP = surfaceTMP
        self.outputSurface = outputSurface
        if surfaceTMP:
            self.contextEnc = ContextEncoderLSTMSurface()
        else:
            self.contextEnc = ContextEncoderLSTM()
        self.ode = APModel(self.S,self.par, self.batch_size,self.dimD, self.metaNet)
        if isPhysics:
            self.ode = APModelPhysics(self.S,self.par, self.batch_size,self.dimD, self.metaNet)
        elif isNeural:
            pass
            #self.ode = APModelNeural(self.S,self.par, self.batch_size,self.dimD, self.metaNet)
        t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
        val=0
        for i in range(375):
            t[i] = val
            val += 0.9
        self.t = t
        self.t.requires_grad=True
        self.save_hyperparameters()
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight,0,0.01)
            nn.init.zeros_(m.bias)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,weight_decay=0.0001)
        #optimizer.add_param_group({'params':self.metaNet.parameters(),'weight_decay':0.1, 'lr':1e-5})
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #optimizer, T_0=t0, T_mult=2, eta_min=lr_min)
        #scheduler=LinearWarmupCosineAnnealingLR(optimizer, 2, 100, warmup_start_lr=0.0)
        

        
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        #print(self.ode.par.grad)
        #print(self.ode.p.grad)
        for p in self.contextEnc.parameters():
            print(p.grad)
        #norms = grad_norm(self.contextEnc, norm_type=2)
        norms = grad_norm(self.metaNet, norm_type=2)
        print(norms)
        self.log_dict(norms,prog_bar=True)
    
    
    
    
    
    def forward(self, UV,k,y=None) -> Any:
        
        
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
        bs, num_k, seq_len, dim=k.size()
        if not self.isRecon:
            print("k in for")
            print(k.shape)
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
       
        
        self.ode.par = out
       
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        if self.outputSurface:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        return TMP
    
    def generateTMP(self,UV,par):
        t0 = torch.tensor(0.0)
                
        for i,t in enumerate(self.t[1:]):
            print(i)        
            options = {}
            options.update({'method': 'Dopri5'})
            options.update({'h': None})
            options.update({'t0': t0})
            options.update({'t1': t})
            options.update({'rtol': 1e-7})
            options.update({'atol': 1e-8})
            options.update({'print_neval': False})
            options.update({'neval_max': 1000000})
            options.update({'t_eval':None})
            options.update({'interpolation_method':'cubic'})
                    
            if i ==0:
                p = par.unsqueeze(1)
                UV=torch.cat((UV,p),dim=1)
                TMP = odesolve_adjoint(self.ode, UV, options = options)
                UV_temp=UV.unsqueeze(1)
                TMP_1=TMP.unsqueeze(1)
                TMP_1 = torch.cat((UV_temp,TMP_1), 1)
            else:
                TMP = TMP[:,:self.dimD*2]
                p = par.unsqueeze(1)
                TMP=torch.cat((TMP,p),dim=1)
                TMP = odesolve_adjoint(self.ode, TMP, options = options)
                TMP_new=TMP.unsqueeze(1)
                TMP_1=torch.cat((TMP_1,TMP_new),1)
            t0 = t 
        
        #print(TMP.shape)
        #TMP=odeint(self.ode, UV, self.t, method='dopri8',adjoint_options={"norm": "seminorm"})
        
        #print("--- %s seconds ---" % (time.time() - tn))
        #print(torch.cuda.memory_allocated(0))
        TMP = TMP_1[:, :, 0:self.dimD]
        return TMP
    
    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        """def hook_fn(module,res, grad):
            print("Gradient:", grad)

        # Register the hook on the linear layer's weights
        
        self.metaNet.register_full_backward_hook(hook_fn)"""
        t0 = time.time()
        
        UV, y, k, a = train_batch #uv=bs*3724, y=bs*375*1862, k=bs*10*375*1862, a=32
        #torch.Size([16, 10, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
            
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        if not self.isRecon:
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
            
        
        """for i in range(num_k):
            temp=k[:,i,:,:]
            
            par += self.contextEnc(temp)
        par = par/num_k  """
        
        
        #print("--- %s seconds ---" % (time.time() - t0))
        #print(par)
        #par=par.squeeze()
        
        #=torch.ones(1862,self.batch_size).to(device)
        #p = p*out
        
        #print(par)
        tn = time.time()
        """for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()"""
        #print("--- %s seconds ---" % (time.time() - t0))
         
        
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        
        """t0 = torch.tensor(0.0)
                
        for i,t in enumerate(self.t[1:]):
                    
            options = {}
            options.update({'method': 'Dopri5'})
            options.update({'h': None})
            options.update({'t0': t0})
            options.update({'t1': t})
            options.update({'rtol': 1e-7})
            options.update({'atol': 1e-8})
            options.update({'print_neval': False})
            options.update({'neval_max': 1000000})
            options.update({'t_eval':None})
            options.update({'interpolation_method':'cubic'})
                    
            if i ==0:
                TMP = odesolve_adjoint(self.ode, UV, options = options)
                UV_temp=UV.unsqueeze(1)
                TMP_1=TMP.unsqueeze(1)
                TMP_1 = torch.cat((UV_temp,TMP_1), 1)
            else:
                TMP = odesolve_adjoint(self.ode, TMP, options = options)
                TMP_new=TMP.unsqueeze(1)
                TMP_1=torch.cat((TMP_1,TMP_new),1)
            t0 = t 
        
        #print(TMP.shape)
        #TMP=odeint(self.ode, UV, self.t, method='dopri8',adjoint_options={"norm": "seminorm"})
        
        #print("--- %s seconds ---" % (time.time() - tn))
        #print(torch.cuda.memory_allocated(0))
        TMP = TMP_1[:, :, 0:self.dimD]"""
        #TMP=odeint(self.ode, UV, self.t, method='dopri5')
        #TMP = TMP[:,:,0:self.dimD]
        #TMP = TMP.permute(1,0,2)
        TMP=self.generateTMP(UV,self.ode.par)
        if self.outputSurface:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
            y = y.permute(0,2,1)
            y = y[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        loss = F.mse_loss(TMP, y) 
        self.log_dict({"train": loss, "par": param_loss},prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        #loss.backward()
        
        
        return loss #+ 100*param_loss
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        t0 = time.time()
        
        UV, y, k, a = train_batch
        
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        if not self.isRecon:
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
        
        """for i in range(num_k):
            temp=k[:,i,:,:]
            
            par += self.contextEnc(temp)
        par = par/num_k  """
        #print("time for context")
        
        """for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()"""
        #print("--- %s seconds ---" % (time.time() - t0))
         
        #print(torch.cuda.memory_allocated(0))
        #print("time for ode")
        #t0 = time.time()
        #self.par.requires_grad = True
        #p = torch.nn.Parameter(p, requires_grad=True)
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        if self.outputSurface:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
            y = y.permute(0,2,1)
            y = y[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        
        loss = F.mse_loss(TMP, y) 
        self.log_dict({"val_loss": loss },prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss 
    
    
    def test_step(self,train_batch, train_idx) -> STEP_OUTPUT:
           
        def get_activation_len_new(tmp):
            bs, time, features = tmp.shape
            start = torch.ones((bs,features))
            end = torch.zeros((bs, features))
            act_len = torch.ones((bs,features))
            for b in range(bs):
                for f in range(features):
                    tm = tmp[b,:,f]
                    grad = tm[..., 1:] - tm[..., :-1]
                    
                    #print("this is grad")
                    #print(grad)
                    #print(torch.max(grad,0))
                    val, index=torch.max(tm,0)
                    val, index = val.item(), index.item()
                    s = 0
                    e = 375
                    for i,val in enumerate(grad):
                        if val>=4.0e-2:
                            s=i
                            break
                    """for i in reversed(range(index)):
                        if tm[i]>val and tm[index]-tm[i]>0.3:
                            j= i - 1
                            
                            if j>=0 and tm[j]>=tm[i]:
                                s = i
                                break
                        val = tm[i]"""
                    for i in range(index,375):
                        if tm[i]<=0:
                            e = i
                            break
                    #print(start,end)
                    start[b][f] = s
                    end[b][f] = e
                    act_len[b][f] = e - s 
            return act_len, start, end    
        t0 = time.time()
        
        UV, y, k, a = train_batch
        act_len_gt, start_gt, end_gt=get_activation_len_new(y)
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        if not self.isRecon:
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
        
        """for i in range(num_k):
            temp=k[:,i,:,:]
            
            par += self.contextEnc(temp)
        par = par/num_k  """
        #print("time for context")
        
        """for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()"""
        #print("--- %s seconds ---" % (time.time() - t0))
         
        #print(torch.cuda.memory_allocated(0))
        #print("time for ode")
        #t0 = time.time()
        #self.par.requires_grad = True
        #p = torch.nn.Parameter(p, requires_grad=True)
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        act_len_model, start_model, end_model=get_activation_len_new(TMP)
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        
        loss = F.mse_loss(TMP, y) 
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        std = torch.std(y-TMP)
        std_param=torch.std(a-self.ode.par)
        loss = F.mse_loss(TMP, y) 
        act_loss = F.mse_loss(act_len_model,act_len_gt)
        start_loss = F.mse_loss(start_model,start_gt)
        end_loss = F.mse_loss(end_model, end_gt)
        self.log_dict({"test": loss ,'param_loss':param_loss,'std':std,'std_param':std_param,'act':act_loss,'start':start_loss, 'end':end_loss},prog_bar=True,)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss 
    

    

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
        #self.fcn = MetaNet1()
        """self.nn = Sequential('x, edge_index, pseudo', [
    (spline_conv.SplineConv(2,1,1,2), 'x, edge_index, pseudo -> x'),
    ReLU(inplace=True),
    (spline_conv.SplineConv(1,1,1,2), 'x, edge_index, pseudo -> x'),
    
    #Linear(64, out_channels),
    ]).to(device)"""
        self.nn = spline_conv.SplineConv(2,1,1,5)
        
        self.act = nn.Tanh()
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
        
            #print(self.expaned_edges.shape, self.expanded_pseudo)
        u = torch.t(y[:,0:self.dimD]) #1862 , bs
            
        v = torch.t(y[:,self.dimD:self.dimD*2])
            
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
                    uv=torch.cat((u1,v1),0)
                else:
                    uv = torch.cat((uv, torch.cat((u1,v1),0)),1)"""
                
                    
                    
            
        out = self.nn(torch.t(uv),self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
            #out = self.nn1(out, self.expaned_edges,torch.t(self.expanded_pseudo.unsqueeze(0)))
        #out = self.act(out)
        #out = torch.nn.Parameter(out)
            
        pde1 = MK + k*u*(1-u)*(u-self.par) - out.view(1862,-1)
        pde2 = -e*(k*u*(u-self.par-1)+v)
        
            #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        return torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))
    

        
    
    
class HybridModel1(L.LightningModule):
    def __init__(self, S, dimD,par, batch_size,edges,pseudo,degree, norm,bias=None, root_weight=None,sufaceIds=None,isPhysics = False, isNeural=False, isRecon= False,surfaceTMP=False):
        super().__init__()
        
        self.S = S
        self.dimD = dimD
        self.par = par
        self.batch_size = batch_size
        
        t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
        val=0
        for i in range(375):
            t[i] = val
            val += 0.9
        self.t = t
        
        self.ode = APModel1(self.S,self.par, self.batch_size,self.dimD,edges,pseudo,degree,norm,bias,root_weight)
        self.contextEnc =  ContextEncoderLSTM()
        self.surfaceIds = sufaceIds
        self.isPhysics = isPhysics
        self.isNeural = isNeural
        self.isRecon = isRecon
        self.isSurfaceTMP = surfaceTMP
        self.save_hyperparameters()
        
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,weight_decay=0.01)
        #optimizer.add_param_group({'params':self.metaNet.parameters(),'weight_decay':0.1, 'lr':1e-5})
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #optimizer, T_0=t0, T_mult=2, eta_min=lr_min)
        #scheduler=LinearWarmupCosineAnnealingLR(optimizer, 1, 100, warmup_start_lr=0.0)
        

        
        return optimizer
    
    """def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.ode, norm_type=2)
        self.log_dict(norms,prog_bar=True)"""
    
    
    def forward(self, UV,k,y=None) -> Any:
        
        
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
        bs, num_k, seq_len, dim=k.size()
        if not self.isRecon:
            print("k in for")
            print(k.shape)
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
       
        
        self.ode.par = out
       
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        if self.isSurfaceTMP:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        return TMP
    
    
    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        
        UV, y, k, a = train_batch
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        k=k.view(-1,seq_len,dim)
        out=self.contextEnc(k)
        out=out.view(bs,-1)
        out=torch.mean(out,dim=1)
        
        #print(par)
        #par=par.squeeze()
        
        self.par = out   
       
        
        self.ode.par = out
        
        
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        if self.isSurfaceTMP:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        loss = F.mse_loss(TMP, y) 
        self.log_dict({"train": loss, "par": param_loss},prog_bar=True,on_step=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss #+ 100*param_loss
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        
        UV, y, k, a = train_batch
        
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            y = y.permute(0,2,1)
            y = y[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        if not self.isRecon:
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
        
        """for i in range(num_k):
            temp=k[:,i,:,:]
            
            par += self.contextEnc(temp)
        par = par/num_k  """
        #print("time for context")
        
        """for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()"""
        #print("--- %s seconds ---" % (time.time() - t0))
         
        #print(torch.cuda.memory_allocated(0))
        #print("time for ode")
        #t0 = time.time()
        #self.par.requires_grad = True
        #p = torch.nn.Parameter(p, requires_grad=True)
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        if self.isSurfaceTMP:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        
        loss = F.mse_loss(TMP, y) 
        self.log_dict({"val_loss": loss },prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss 
        
    
    def test_step(self,train_batch, train_idx) -> STEP_OUTPUT:
        UV, y, k, a = train_batch
        
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        if not self.isRecon:
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k)
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
        
        """for i in range(num_k):
            temp=k[:,i,:,:]
            
            par += self.contextEnc(temp)
        par = par/num_k  """
        #print("time for context")
        
        """for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()"""
        #print("--- %s seconds ---" % (time.time() - t0))
         
        #print(torch.cuda.memory_allocated(0))
        #print("time for ode")
        #t0 = time.time()
        #self.par.requires_grad = True
        #p = torch.nn.Parameter(p, requires_grad=True)
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        
        loss = F.mse_loss(TMP, y) 
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        std = torch.std(y-TMP)
        std_param=torch.std(a-self.ode.par)
        loss = F.mse_loss(TMP, y) 
        
        self.log_dict({"test": loss ,'param_loss':param_loss,'std':std,'std_param':std_param},prog_bar=True,)
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
    
class MetaNet3(nn.Module):
    def __init__(self):
        super(MetaNet3, self).__init__()
        self.fc1= nn.Sequential(nn.Linear(1862, 64) ,nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64,128),nn.ReLU())
        #self.fc3 = nn.Sequential(nn.Linear(64,128),nn.SiLU())
        self.fc3 = nn.Sequential(nn.Linear(128,1862),nn.Tanh())
        
        
        
    def forward(self, x) -> Any:
        
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        #out = self.fc4(out)
        
        return out
    
    
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
    
class MetaNet5(nn.Module):             #40         #20
    def __init__(self,input_dim=1862, seq_l=375,mid_dim=128, mid_1=32,out_dim=1862, mid_d=64):
        super(MetaNet5, self).__init__()
        self.l1 = nn.LSTM(input_dim,mid_dim,batch_first=True)
        self.l2 = nn.LSTM(mid_dim,mid_1,batch_first=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ll1 = nn.Linear(mid_1*seq_l, mid_d)
        self.ll2 = nn.Linear(mid_d, out_dim)
        self.silu = nn.SiLU()
        
    
    def forward(self, x) -> Any:
        batch, _, _ = x.size()
        
        out, h1 = self.l1(x)
        
        out = self.relu(out)
        out, h2 = self.l2(out)
        
        out = out.contiguous().view(batch,-1)
        out = self.ll1(out)
        out=self.relu(out)
        out = self.ll2(out)
        out = self.silu(out)
        
        
        
        return out

class OnlyNN(L.LightningModule):
    def __init__(self, S, dimD,par, batch_size, isPhysics = False, isNeural=False):
        super().__init__()
        #self.metaNet = MetaNet2(batch_size).to(device)
        #self.metaNet.apply(self.init_weights)
        self.metaNet = MetaNet4()
        self.S = S
        self.dimD = dimD
        self.par = par
        self.batch_size = batch_size
        
        self.contextEnc = ContextEncoderLSTM()
        self.ode = APModel2(self.S,self.par, self.batch_size,self.dimD, self.metaNet).to(device)
        self.solver = NeuralODE(self.ode,solver='dopri5',sensitivity='adjoint').to(device)
            #self.ode = APModelNeural(self.S,self.par, self.batch_size,self.dimD, self.metaNet)
        t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
        val=0
        for i in range(375):
            t[i] = val
            val += 0.9
        self.t = t
        self.t.requires_grad=True
        self.save_hyperparameters()
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight,0,0.01)
            nn.init.zeros_(m.bias)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,weight_decay=0.0001)
        #optimizer.add_param_group({'params':self.metaNet.parameters(),'weight_decay':0.1, 'lr':1e-5})
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #optimizer, T_0=t0, T_mult=2, eta_min=lr_min)
        #scheduler=LinearWarmupCosineAnnealingLR(optimizer, 2, 100, warmup_start_lr=0.0)
        

        
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        #print(self.ode.par.grad)
        #print(self.ode.p.grad)
        """for p in self.metaNet.meta_net.parameters():
            print(p.grad)
        norms = grad_norm(self.contextEnc, norm_type=2)
        print(norms)
        self.log_dict(norms,prog_bar=True)"""
    
    
    
    
    
    def forward(self, UV,k) -> Any:
        bs, num_k, seq_len, dim=k.size()
        #uv=bs*3724, y=bs*375*1862, k=bs*10*375*1862, a=32
        #torch.Size([16, 10, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        UV = UV[:,0:self.dimD]
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        """k=k.view(-1,seq_len,dim)
        out=self.contextEnc(k)
        out=out.view(bs,-1)
        out=torch.mean(out,dim=1)
        
       
        
        a=torch.ones(self.batch_size)
        self.ode.par = a*torch.tensor(0.1)"""
        #self.metaNet.set_param(out)
       
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        #TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        return TMP
    
    
    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        
        UV, y, k, a = train_batch #uv=bs*3724, y=bs*375*1862, k=bs*10*375*1862, a=32
        #torch.Size([16, 10, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        UV = UV[:,0:self.dimD]
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        """ k=k.view(-1,seq_len,dim)
        #TODO uncomment these
        out=self.contextEnc(k)
        out=out.view(bs,-1)
        out=torch.mean(out,dim=1)
        
       
        
        
        self.ode.par = out
        self.metaNet.set_param(out)"""
        _,TMP=self.solver(UV,self.t)
        #print(x.shape, y.shape)
        #print(TMP.shape,y.shape)
        #TMP=odeint(self.ode, UV, self.t, method='dopri5')
        #TMP = TMP[:,:,0:self.dimD]
        #TMP = TMP.permute(1,0,2)
        y = y.permute(1,0,2)
        #pri nt(a,par)
        #param_loss = F.mse_loss(a, out)
        print(TMP.shape, y.shape)
        loss = F.mse_loss(TMP, y)
        #self.log_dict({"train": loss, "par": param_loss},prog_bar=True)
        self.log_dict({"train": loss},prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss #+ 100*param_loss
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        UV, y, k, a = train_batch #uv=bs*3724, y=bs*375*1862, k=bs*10*375*1862, a=32
        #torch.Size([16, 10, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        UV = UV[:,0:self.dimD]
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        k=k.view(-1,seq_len,dim)
        #TODO uncomment these
        out=self.contextEnc(k)
        out=out.view(bs,-1)
        out=torch.mean(out,dim=1)
        
       
        
        #TODO change this
        self.ode.par = out
        self.metaNet.set_param(out)
       
        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        #TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        
        #pri nt(a,par)
        #param_loss = F.mse_loss(a, out)
        loss = F.mse_loss(TMP, y)
        self.log_dict({"val_loss": loss },prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss 
    
    
    def test_step(self,train_batch, train_idx) -> STEP_OUTPUT:
        t0 = time.time()
        
        UV, y, k, a = train_batch
        #torch.Size([16, 1, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
        k=k.view(-1,seq_len,dim)
        out=self.contextEnc(k)
        out=out.view(bs,-1)
        out=torch.mean(out,dim=1)
        
        """for i in range(num_k):
            temp=k[:,i,:,:]
            
            par += self.contextEnc(temp)
        par = par/num_k  """
        #print("time for context")
        
        """for i in range(self.batch_size):
            #print(i)
            #print(self.batch_size)
            temp     = torch.full((1862,1),par[i].item())
            p[:,i]  = temp.squeeze()"""
        #print("--- %s seconds ---" % (time.time() - t0))
         
        #print(torch.cuda.memory_allocated(0))
        #print("time for ode")
        #t0 = time.time()
        #self.par.requires_grad = True
        #p = torch.nn.Parameter(p, requires_grad=True)
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        

        TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        
        #print(a,par)
        param_loss = F.mse_loss(a, out)
        loss = F.mse_loss(TMP, y) 
        self.log_dict({"test": loss ,'param_loss':param_loss},prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss 
