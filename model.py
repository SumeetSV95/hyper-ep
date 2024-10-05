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
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
#from torch_spline_conv import spline_conv
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
#from torch_geometric.nn.conv import spline_conv
from torch.nn import Linear, ReLU,Tanh
#from torch_geometric.nn import Sequential
from lightning.pytorch.utilities import grad_norm
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#import pytorch_warmup as warmup
from torch.optim.lr_scheduler import ExponentialLR
from scipy.integrate import solve_ivp, LSODA
#import torchode as to
from TorchDiffEqPack import  odesolve_adjoint
#from ignite.handlers import create_lr_scheduler_with_warmup
#from torchdyn.core import NeuralODE
#from torch_geometric.nn.conv import GCNConv
import torch.nn.init as init
import sys
import scipy.io
sys.path.append('/home/sv6234/ECGI/ECGI')
from modules import *
#from IRDM.interpolated_torchdiffeq import odeint_chebyshev_func
from loss.dilate_loss import dilate_loss
import pysdtw
from collections import defaultdict
import random
from utils import get_activation_len_new
from functools import partial
import pandas as pd
import functorch
import torch
import torch.autograd.profiler as profiler


device = 'cuda:0'

        


    
class HybridModel(L.LightningModule):
    def __init__(self, S, dimD,par, batch_size, sufaceIds=None,isPhysics = False, isNeural=False, isRecon= False,surfaceTMP=False,outputSurface = False,adjoint=False,EGM_in=False,EGM_out=False,dilate_loss=False,use_resnet=False,seed=None):
        super().__init__()
        self.metaNet = MetaNet_new() # chaneg this to MetaNet_new
        #self.metaNet = MetaNet_new_v1()
        #self.metaNet = MetaNet_new_v2()
        #self.metaNet.apply(self.init_weights)#this change
        self.S = S
        self.dimD = dimD
        self.par = par
        self.batch_size = batch_size
        self.isRecon = isRecon
        self.surfaceIds = sufaceIds
        self.isSurfaceTMP = surfaceTMP
        self.outputSurface = outputSurface
        self.adjoint = adjoint
        self.EGM_in = EGM_in
        self.EGM_out = EGM_out
        self.H = None
        self.dilate_loss = dilate_loss
        if EGM_out:
            mat = scipy.io.loadmat('/home/sv6234/ECGI/H_Trans.mat')
            mat = mat['H']
            H=torch.from_numpy(mat).float().to(device)
            
            self.H=H
            #self.H.requires_grad=False
        if surfaceTMP:
            self.contextEnc = ResNet1D(ResidualBlock, [2, 2, 2, 2], input_channels=477, num_classes=1)
        elif self.EGM_in:
            self.contextEnc = ResNet1D(ResidualBlock, [2, 2, 2, 2], input_channels=396, num_classes=1)  #ContextEncoderLSTM_EGM()
        else:
            if use_resnet:
                #print("Using ResNet")
                self.contextEnc = ResNet1D(ResidualBlock, [2, 2, 2, 2], input_channels=1119, num_classes=1)
            else:
                self.contextEnc = ContextEncoderLSTM()
        #self.ode = APModel(self.S,self.par, self.batch_size,self.dimD, self.metaNet,self.adjoint)
        self.ode = APModel_uv(self.S,self.par, self.batch_size,self.dimD, self.metaNet,self.adjoint)
        if isPhysics:
            self.ode = APModelPhysics(self.S,self.par, self.batch_size,self.dimD, self.metaNet)
        elif isNeural:
            pass
            #self.ode = APModelNeural(self.S,self.par, self.batch_size,self.dimD, self.metaNet)
        """t=torch.from_numpy(np.zeros(375)).to(torch.float32).to(device)
        val=0
        for i in range(375):
            t[i] = val
            val += 0.9
        self.t = t"""
        self.t=torch.from_numpy(np.linspace(0,350,117)).to(device)
        self.t.requires_grad=True
        self.alpha = 0.6#0.0005 #0.00005 0.6 for other model.
        self.gamma = 0.001
        self.c = 0.001
        fun = pysdtw.distance.pairwise_l2_squared_exact
        self.sdtw = pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=True)
        #self.solver = partial(odeint_chebyshev_func, n_nodes=375, method='dopri5',t =torch.tensor([0.0,336.6]).to(device))
        
        self.weights = defaultdict(lambda: torch.tensor(0.25))  # Initialize group weights
        self.eta = 0.1
        if seed is not None:
            self.set_seed(seed)
        self.save_hyperparameters()
    
    def set_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight,0,0.01)
            nn.init.zeros_(m.bias)
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        #optimizer = torch.optim.Adam(self.parameters(),lr=1e-2,weight_decay=0.0001)
        optimizer = torch.optim.Adam([{'params':self.ode.parameters(),'weight_decay':0.0001, 'lr':1e-3},#3,#-2 for the main EGM param is 5e-3
                                      {'params':self.contextEnc.parameters(),'weight_decay':0.0001, 'lr':1e-2}])#4#-3
        #optimizer.add_param_group({'params':self.metaNet.parameters(),'weight_decay':0.1, 'lr':1e-5})
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #optimizer, T_0=t0, T_mult=2, eta_min=lr_min)
        #scheduler=LinearWarmupCosineAnnealingLR(optimizer, 2, 100, warmup_start_lr=0.0)
        

        
        return optimizer
    
    """def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        #print(self.ode.par.grad)
        #print(self.ode.p.grad)
        for p in self.contextEnc.parameters():
            print(p.grad)
        #norms = grad_norm(self.contextEnc, norm_type=2)
        norms = grad_norm(self.metaNet, norm_type=2)
        print(norms)
        self.log_dict(norms,prog_bar=True)"""
    
    
    
    
    
    def forward(self, UV,k,y=None) -> Any:
        
        
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
        bs, num_k, seq_len, dim=k.size()
        if not self.isRecon:
            k=k.view(-1,seq_len,dim)
            out=self.contextEnc(k.permute(0,2,1))
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
       
        
        self.ode.par = out
       
        if self.adjoint:
            TMP=self.generateTMP(UV,self.ode.par)
        else:
            TMP=odeint(self.ode, UV, self.t, method='dopri5')
            
            #TMP=self.solver(self.ode,UV,self.t,method='dopri5')
            TMP = TMP[:,:,0:self.dimD]
            TMP = TMP.permute(1,0,2)       

        
    
        
        return TMP
    
    def generateTMP(self,UV,par):
        t0 = torch.tensor(0.0)
                
        for i,t in enumerate(self.t[1:]):
            #print(i)        
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
    def plot_tmp(self, TMP, y, a, out, name):
        # Detach tensors from computation graph and move to CPU for plotting
        y_detached = y.detach().cpu().numpy()
        TMP_detached = TMP.detach().cpu().numpy()
        a_detached = a.detach().cpu().numpy()
        out_detached = out.detach().cpu().numpy()
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 3))
        
        ax1.plot(y_detached[0, :, 0], label='a gt' + str(a_detached[0]))
        ax1.plot(TMP_detached[0, :, 0], label='a' + str(out_detached[0]))
        ax1.legend()
        
        ax2.plot(y_detached[2, :, 0], label='a gt' + str(a_detached[2]))
        ax2.plot(TMP_detached[2, :, 0], label='a' + str(out_detached[2]))
        ax2.legend()
        
        ax3.plot(y_detached[3, :, 0], label='a gt' + str(a_detached[3]))
        ax3.plot(TMP_detached[3, :, 0], label='a' + str(out_detached[3]))
        ax3.legend()
        
        # Save plot to file
        plt.savefig(name)
        plt.close()

        # Ensure garbage collection of the figure
        fig.clf()
        plt.close(fig)
    
    def get_loss_wrt_a(self, TMP, y, a):
        """
        Compute the loss for each group, ensuring that the loss requires gradients.
        """
        loss_dict = defaultdict(list)
        
        # Loop through each batch sample
        for i in range(self.batch_size):
            # Check if each TMP[i] requires gradients
            assert TMP[i].requires_grad, f"TMP[{i}] does not require gradients"
            
            # Compute loss for this sample
            loss = F.mse_loss(TMP[i, :, 0], y[i, :, 0])
            
            # Append loss to the appropriate group based on `a`
            loss_dict[a[i].item()].append(loss)

        # Average the losses for each group
        for key in loss_dict:
            group_loss = torch.stack(loss_dict[key]).mean()
            assert group_loss.requires_grad, f"group_loss for group {key} does not require gradients"
            loss_dict[key] = group_loss

        return loss_dict



    
    def update_loss_dict(self, loss_dict,train=True):
        if train:
            for key in loss_dict:
                if key not in self.train_loss:
                    self.train_loss[key] = loss_dict[key]
                else:
                    self.train_loss[key]+= loss_dict[key]
        else:
            for key in loss_dict:
                if key not in self.val_loss:
                    self.val_loss[key] = loss_dict[key]
                else:
                    self.val_loss[key]+= loss_dict[key]
                    
    def reset(self):
        self.train_loss = {}
        self.train_grads = defaultdict(None)
        self.val_loss = {}
        self.val_grads = defaultdict(None)
        
    def compute_group_gradients(self, loss_dict, params):
        """
        Manually compute gradients for all group losses.
        """
        group_grads = {}

        # Loop through each group and compute gradients
        for group_loss_key, group_loss in loss_dict.items():
            assert group_loss.requires_grad, f"group_loss for group {group_loss_key} does not require gradients"
            
            # Compute the gradient for this group's loss
            grads = torch.autograd.grad(group_loss, params, retain_graph=True)
            group_grads[group_loss_key] = grads

        return group_grads


    def update_weights(self, train_loss, train_grads):
        """
        Update the task weights using CGD-style update with cosine similarity between gradients
        and group losses. This follows the CGD process of multiplicative weight update.
        """
        weight_updates = defaultdict(lambda: torch.tensor(1.0))  # Initialize weight updates with 1

        # Compute the weight update increment based on the cosine similarity between gradients and group losses
        for a_val_1 in train_loss:
            # Compute GIGSUM as a sum of group losses and cosine similarities between their gradients
            gigsum = torch.sum(torch.stack(
                [torch.sqrt(train_loss[a_val_1] * train_loss[a_val_2]) *
                torch.nn.functional.cosine_similarity(train_grads[a_val_1], train_grads[a_val_2], dim=0)
                for a_val_2 in train_loss]))

            # Apply CGD's exponential scaling for the weight update
            weight_updates[a_val_1] = self.weights[a_val_1] * torch.exp(self.eta * gigsum.detach())

        # Normalize weights to ensure they sum to 1 after the update
        prob_sum = sum(torch.sum(p) for p in weight_updates.values() if isinstance(p, torch.Tensor))
        for task in weight_updates.keys():
            self.weights[task] = weight_updates[task] / prob_sum  # Normalize to sum to 1
            self.weights[task] = torch.clamp(self.weights[task], min=1e-5)  # Ensure non-negativity


    def get_weighted_loss(self, TMP, y, weights, a):
        """
        Compute the weighted loss using group weights.
        """
        # Initialize loss without requires_grad=True; let it accumulate naturally
        loss = torch.tensor(0.0, device=TMP.device)  # Ensure it's on the same device as TMP
        
        # Get group-wise losses
        loss_dict = self.get_loss_wrt_a(TMP, y, a)
        
        # Accumulate weighted losses
        for a_val in weights:
            loss += weights[a_val] * loss_dict[a_val]
        
        return loss
     
                       
    def training_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        """def hook_fn(module,res, grad):
            print("Gradient:", grad)

        # Register the hook on the linear layer's weights
        
        self.metaNet.register_full_backward_hook(hook_fn)"""
        t0 = time.time()
        
        UV, y, k, a, _ = train_batch #uv=bs*3724, y=bs*375*1862, k=bs*10*375*1862, a=32
        #torch.Size([16, 10, 375, 1862]) torch.Size([16])
        #k =k.squeeze(1)
        #print(UV.shape,y.shape,k.shape,a.shape)
        if self.isSurfaceTMP:
            k=k.permute(0,1,3,2)
            k = k[:,:,self.surfaceIds,:].squeeze().permute(0,1,3,2)
            
            
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        with profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof_resnet:
            with profiler.record_function("ResNet_Evaluation"):
                if not self.isRecon:
                    k=k.view(-1,seq_len,dim)
                    if isinstance(self.contextEnc, (ContextEncoderLSTM_EGM,ContextEncoderLSTMSurface,ContextEncoderLSTM)):
                        out=self.contextEnc(k)# remove this for LSTM
                    else:
                        #print(k.shape)
                        out=self.contextEnc(k.permute(0,2,1))
                    out=out.view(bs,-1)
                    out=torch.mean(out,dim=1)
                else:
                    out=self.contextEnc(y)
                    out=out.squeeze()
        print(prof_resnet.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        
       
        tn = time.time()
       
        
         
        
        
        self.ode.par = out
        with torch.autograd.profiler.profile(record_shapes=True, use_cuda=True, profile_memory=True) as prof:
            # Your code to profile

            with profiler.record_function("ODE_Solver"):
                if not self.adjoint:
                    #print(self.t)
                    
                    TMP=odeint(self.ode, UV, self.t, method='dopri5')
                    self.ode.count = 0
                    #TMP=self.solver(self.ode,UV)
                    TMP = TMP[:,:,0:self.dimD]
                    TMP = TMP.permute(1,0,2)
                else:
                    TMP=self.generateTMP(UV,self.ode.par)
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        loss1 = torch.tensor([0.0])    
        param_loss = F.mse_loss(out, a)
        loss_1 = F.mse_loss(TMP, y)
        print("--- %s seconds ---" % (time.time() - t0))
        if self.outputSurface:
            loss1 = F.mse_loss(TMP.clone(), y.clone())
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
            y = y.permute(0,2,1)
            y = y[:,self.surfaceIds,:].squeeze().permute(0,2,1)
            #self.plot_tmp(TMP,y,a,out,'TMP_sur_in_sur_out.png')
            if self.dilate_loss:
                loss,l_shape,l_temporal = dilate_loss(TMP,y,self.alpha,self.gamma,device)
            else:
                #current loss
                #loss = self.get_weighted_loss(TMP, y, self.weights, a)
                loss = F.mse_loss(TMP, y)
                # Compute group-wise losses
                """loss_dict = self.get_loss_wrt_a(TMP, y, a)

                # Vectorized gradient computation
                params = list(self.parameters())
                for param in self.parameters():
                    if param.grad is None:
                        print(f"Parameter {param} does not have gradients")
                grads = torch.autograd.grad(loss, params, retain_graph=True)
                print('these are gradients')
                print(grads)    
                group_grads = self.compute_group_gradients(loss_dict, params)
                print('these are gradients')
                print(group_grads)
                # Incrementally update weights based on gradients, scaled by step size
                self.update_weights(loss_dict, group_grads)
                print(self.weights)
                    #TODO not accumulateing the gradients anymore
                    # Accumulate gradients for each parameter
                    #self.train_grads[a_val] = [accum + grad.detach() for accum, grad in zip(self.train_grads[a_val], gradients)]
                for key in loss_dict:
                    loss_dict[key] = loss_dict[key].detach()
                
                    
                #no need to update the loss dict as we are not accumulating the gradients
                #TODO write a function to update the weights for the next batch
                
                #self.update_loss_dict(loss_dict,train=True)
                print("--- %s seconds ---" % (time.time() - t0))"""
            
        #print(a,par)
        elif self.EGM_out:
            EGM = torch.zeros((bs,375,396)).to(device)
            EGM_GT = torch.zeros((bs,375,396)).to(device)
            for i in range(bs):
                EGM[i] = torch.matmul(self.H,TMP[i].permute(1,0)).permute(1,0)
                EGM_GT[i] = torch.matmul(self.H,y[i].permute(1,0)).permute(1,0)
            #EGM = torch.matmul(self.H,TMP.permute(0,2,1)).permute(0,2,1)
            #EGM_GT = torch.matmul(self.H,y.permute(0,2,1)).permute(0,2,1)
            if not self.dilate_loss:
                loss1 = F.mse_loss(EGM,EGM_GT)
                res= self.sdtw(EGM,EGM_GT) - (1/2)*(self.sdtw(EGM,EGM)+self.sdtw(EGM_GT,EGM_GT))
                dt_loss=res.mean()
                loss = loss1 + dt_loss
            else:
                
                EGM_reduce_res=EGM[:,:,:]
                
                EGM_GT_reduce_res=EGM_GT[:,:,:]
                loss,l_shape,l_temporal = dilate_loss(EGM_reduce_res,EGM_GT_reduce_res,self.alpha,self.gamma,device)
            a_copy = a.clone().detach().cpu().numpy()
            out_copy = out.clone().detach().cpu().numpy()
            fig, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4, figsize=(24, 3))
            ax1.plot(y[0,:,0].detach().cpu().numpy(),label='a gt'+ str(a_copy[0].item()))
            
            ax1.plot(TMP[0,:,0].detach().cpu().numpy(),label='a'+str(out_copy[0].item()))
            ax1.legend()
            ax2.plot(EGM_GT[0,:,0].detach().cpu().numpy())
            ax2.plot(EGM[0,:,0].detach().cpu().numpy())
            
            ax3.plot(y[1,:,0].detach().cpu().numpy(),label='a gt'+ str(a_copy[1].item()))
            
            ax3.plot(TMP[1,:,0].detach().cpu().numpy(),label='a'+str(out_copy[1].item()))
            ax3.legend()
            ax4.plot(EGM_GT[1,:,0].detach().cpu().numpy())
            ax4.plot(EGM[1,:,0].detach().cpu().numpy())
            plt.savefig('EGM_2.png')
            plt.close() 
            
        else:
            #self.plot_tmp(TMP,y,a,out,'TMP_sur_in_full_out_1.png')
            if self.dilate_loss:
                
                loss,l_shape,l_temporal = dilate_loss(TMP,y,self.alpha,self.gamma,device)
            else:
                #loss1 = F.mse_loss(EGM,EGM_GT)
                
                loss = F.mse_loss(TMP, y)
        if self.dilate_loss:
            self.log_dict({"train": loss, "par": param_loss,'AP_loss':loss_1,'l_shape':l_shape,'l_temporal':l_temporal},prog_bar=True)
        else:    
            self.log_dict({"train": loss, "par": param_loss,'3D signal':loss_1,'AP_loss':loss_1},prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        #loss.backward()
        
        
        return loss #+ 100*param_loss
    """def on_train_epoch_end(self):
        # Average the accumulated gradients
        num_steps = 66  # Number of steps in the epoch
        for a_val in self.train_grads:
            self.train_grads[a_val] = torch.cat([accum / num_steps for accum in self.train_grads[a_val]])
            
        for key in self.train_loss:
            self.train_loss[key] = self.train_loss[key] / num_steps"""
        
        
    
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        t0 = time.time()
        
        UV, y, k, a,_ = train_batch
        
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
            if isinstance(self.contextEnc, (ContextEncoderLSTM_EGM,ContextEncoderLSTMSurface,ContextEncoderLSTM)):
                out=self.contextEnc(k)
            else:
                out=self.contextEnc(k.permute(0,2,1))
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
        if self.adjoint:
            TMP=self.generateTMP(UV,self.ode.par)
        else:
            TMP=odeint(self.ode, UV, self.t, method='dopri5')
            #TMP=self.solver(self.ode,UV,self.t,method='dopri5')
            TMP = TMP[:,:,0:self.dimD]
            TMP = TMP.permute(1,0,2)
        if self.outputSurface:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
            y = y.permute(0,2,1)
            y = y[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        
        #print(a,par)
        """param_loss = F.mse_loss(a, out)
        loss_dict=self.get_loss_wrt_a(TMP, y, a)
        for a_val in loss_dict:
            loss = loss_dict[a_val]
            gradients=torch.autograd.grad(loss, self.parameters())
            if self.val_grads[a_val] is None:
                self.val_grads[a_val] = [torch.zeros_like(grad) for grad in gradients]

            # Accumulate gradients for each parameter
            self.val_grads[a_val] = [accum + grad.detach() for accum, grad in zip(self.val_grads[a_val], gradients)]
        for key in loss_dict:
            loss_dict[key] = loss_dict[key].detach()
        self.update_loss_dict(loss_dict,train=False)"""
        
        loss = F.mse_loss(TMP, y) 
        #loss,l_shape,l_temporal = dilate_loss(TMP,y,self.alpha,self.gamma,device)
        self.log_dict({"val_loss": loss },prog_bar=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss 
    
    """def on_validation_epoch_end(self):
        num_steps = 66  # Number of steps in the epoch
        for a_val in self.val_grads:
            self.val_grads[a_val] = torch.cat([accum / num_steps for accum in self.val_grads[a_val]])
            
        for key in self.val_loss:
            self.val_loss[key] = self.val_loss[key] / num_steps
            
        for a_val_train in self.train_loss:
            gigsum = torch.sum(torch.stack(
                [torch.sqrt(self.train_loss[a_val_train] * self.val_loss[a_val_validation]) * torch.nn.functional.cosine_similarity(
                    self.train_grads[a_val_train] , self.val_grads[a_val_validation], dim=0) for a_val_validation in (self.val_loss)]))
            self.weights[a_val_train] = 1 / 4 * torch.exp(self.eta * gigsum)
            #self.gi_gsum_per_epoch[task].append(gigsum.item())
            prob_sum = sum(torch.sum(p) for p in self.weights.values() if isinstance(p, torch.Tensor))
            for task in self.weights.keys():
                self.weights[task] = self.weights[task] / prob_sum
        self.reset()"""
    
    def test_step(self,train_batch, train_idx) -> STEP_OUTPUT:
        internal_nodes = []
        for i in range(self.dimD):
             if i not in self.surfaceIds:
                 internal_nodes.append(i)  
        torch.tensor(internal_nodes).to(device)
        t0 = time.time()
        
        UV, y, k, a ,_= train_batch
        print(UV.shape,y.shape,k.shape,a.shape)
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
            out=self.contextEnc(k.permute(0,2,1))
            out=out.view(bs,-1)
            out=torch.mean(out,dim=1)
        else:
            out=self.contextEnc(y)
            out=out.squeeze()
        
        
        
        self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        
        
        
        if self.adjoint:
            TMP=self.generateTMP(UV,self.ode.par)
        else:
            TMP=odeint(self.ode, UV, self.t, method='dopri5')
            #TMP=self.solver(self.ode,UV,self.t,method='dopri5')
            TMP = TMP[:,:,0:self.dimD]
            TMP = TMP.permute(1,0,2)
        act_len_model, start_model, end_model=get_activation_len_new(TMP)
        model_act_dict = defaultdict(list)
        gt_label_act_dict = defaultdict(list)
        gt_tmp_dict_internal = defaultdict(list)
        gt_tmp_dict_surface = defaultdict(list)
        model_tmp_dict_internal = defaultdict(list)
        model_tmp_dict_surface = defaultdict(list)
        param_loss_dict = defaultdict(list)
        TMP_surface = TMP.permute(0,2,1)
        TMP_surface = TMP_surface[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        y_surface = y.permute(0,2,1)
        y_surface = y_surface[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        surface_loss = F.mse_loss(TMP_surface,y_surface)
        TMP_internal = TMP.permute(0,2,1)
        TMP_internal = TMP_internal[:,internal_nodes,:].squeeze().permute(0,2,1)
        y_internal = y.permute(0,2,1)
        y_internal = y_internal[:,internal_nodes,:].squeeze().permute(0,2,1)
        for i in range(self.batch_size):
            model_act_dict[a[i].item()].append(end_model[i,:])
            gt_label_act_dict[a[i].item()].append(end_gt[i,:])
            
            model_tmp_dict_internal[a[i].item()].append(TMP_internal[i,:,:])
            model_tmp_dict_surface[a[i].item()].append(TMP_surface[i,:,:])
            gt_tmp_dict_surface[a[i].item()].append(y_surface[i,:,:])
            gt_tmp_dict_internal[a[i].item()].append(y_internal[i,:,:])
            param_loss_dict[a[i].item()].append(out[i].item())
        act_loss_dict = {}
        tmp_loss_dict = {}
        param_loss_final = {}
        for key in model_act_dict:
            model_vals = torch.stack(model_act_dict[key])
            gt_vals = torch.stack(gt_label_act_dict[key])
            model_tmp_vals_internal = torch.stack(model_tmp_dict_internal[key])
            model_tmp_vals_surface = torch.stack(model_tmp_dict_surface[key])
            gt_tmp_vals_internal = torch.stack(gt_tmp_dict_internal[key])
            gt_tmp_vals_surface = torch.stack(gt_tmp_dict_surface[key])
            act_loss_dict[str(round(key,4))+'_act_loss'] = F.l1_loss(model_vals,gt_vals)
            act_loss_dict[str(round(key,4))+'_act_std'] = torch.std(model_vals-gt_vals)
            tmp_loss_dict[str(round(key,4))+'_tmp_loss_internal'] = F.mse_loss(model_tmp_vals_internal,gt_tmp_vals_internal)
            tmp_loss_dict[str(round(key,4))+'_tmp_loss_surface'] = F.mse_loss(model_tmp_vals_surface,gt_tmp_vals_surface)
            tmp_loss_dict[str(round(key,4))+'_tmp_std_internal'] = torch.std(model_tmp_vals_internal-gt_tmp_vals_internal)
            tmp_loss_dict[str(round(key,4))+'_tmp_std_surface'] = torch.std(model_tmp_vals_surface-gt_tmp_vals_surface)
            param_tensor=torch.tensor(param_loss_dict[key])
            gt_tensor = torch.full(param_tensor.size(),key)
            param_loss_final[str(round(key,4))+'_param_loss'] = F.mse_loss(torch.tensor(param_loss_dict[key]),gt_tensor)
            
            
        param_loss = F.mse_loss(a, out)
        
        loss = F.mse_loss(TMP, y)
        label_dict = {} 
        for i in range(self.batch_size):
            if a[i].item() not in label_dict:
                label_dict[a[i].item()] = [self.ode.par[i].item()]
            else:
                label_dict[a[i].item()].append(self.ode.par[i].item())
        #print(a,par)
        label_stats = {}
        for key in label_dict:
            arr=np.array(label_dict[key])
            mean = np.mean(arr)
            std = np.std(arr)
            label_stats[str(key)+'_mean'] = mean
            label_stats[str(key)+'_std'] = std
        #print(label_stats)
        param_loss = F.mse_loss(a, out)
        std = torch.std(y-TMP)
        std_param=torch.std(a-self.ode.par)
        loss = F.mse_loss(TMP, y) 
        act_loss = F.l1_loss (act_len_model,act_len_gt)
        act_std = torch.std(act_len_model-act_len_gt)
        start_loss = F.l1_loss(start_model,start_gt)
        start_std = torch.std(start_model-start_gt)
        end_loss = F.l1_loss(end_model, end_gt)
        end_std = torch.std(end_model-end_gt)
        TMP_surface = TMP.permute(0,2,1)
        TMP_surface = TMP_surface[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        y_surface = y.permute(0,2,1)
        y_surface = y_surface[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        surface_loss = F.mse_loss(TMP_surface,y_surface)
        TMP_internal = TMP.permute(0,2,1)
        TMP_internal = TMP_internal[:,internal_nodes,:].squeeze().permute(0,2,1)
        y_internal = y.permute(0,2,1)
        y_internal = y_internal[:,internal_nodes,:].squeeze().permute(0,2,1)
        internal_loss = F.mse_loss(TMP_internal,y_internal)
        log_dict={"test": loss ,"surface_loss":surface_loss,"internal_loss":internal_loss,'param_loss':param_loss,'std':std,'std_param':std_param,'act':act_loss,'start':start_loss, 'end':end_loss,'act_std':act_std,'start_std':start_std,'end_std':end_std}
        log_dict.update(label_stats)
        log_dict.update(act_loss_dict)
        log_dict.update(tmp_loss_dict)
        log_dict.update(param_loss_final)
        #print(log_dict)
        self.log_dict(log_dict,prog_bar=True,)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        # Save outputs in-memory
        if not hasattr(self, 'test_outputs'):
            self.test_outputs = []
        self.test_outputs.append(log_dict)
        
        
        
        return loss
    
    def on_test_epoch_end(self):
        # Aggregate results
        aggregated_results = defaultdict(list)
        for output in self.test_outputs:
            for key, value in output.items():
                aggregated_results[key].append(value)

        # Compute averages
        averaged_results = {key: torch.mean(torch.tensor(value)) for key, value in aggregated_results.items()}

        # Create DataFrame from averaged results
        averaged_df = pd.DataFrame.from_dict(averaged_results, orient='index', columns=['Value'])

        # Split DataFrame into main table and new columns table
        main_table_columns = ["test", "param_loss", "std", "std_param", "act", "start", "end", "act_std", "start_std", "end_std"]
        main_table_df = averaged_df.loc[main_table_columns]
        new_columns_df = averaged_df.drop(main_table_columns)

        # Split new columns table into act_loss, param_loss
        act_loss_columns = [col for col in new_columns_df.index if '_act_loss' in col]
        param_loss_columns = [col for col in new_columns_df.index if '_param_loss' in col]

        act_loss_df = new_columns_df.loc[act_loss_columns]
        param_loss_df = new_columns_df.loc[param_loss_columns]

        # Log the averaged results
        self.log_dict(averaged_results, prog_bar=True)

        # Save DataFrames as instance attributes for later use
        self.main_table_df = main_table_df
        self.act_loss_df = act_loss_df
        self.param_loss_df = param_loss_df 
    


        
    
    
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
        
        #optimizer = torch.optim.Adam(self.parameters(),lr=1e-3,weight_decay=0.0001) 
        optimizer = torch.optim.Adam([{'params':self.ode.parameters(),'weight_decay':0.0001, 'lr':1e-2},
                                      {'params':self.contextEnc.parameters(),'weight_decay':0.0001, 'lr':1e-3}])
        #optimizer.add_param_group({'params':self.metaNet.parameters(),'weight_decay':0.1, 'lr':1e-5})
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #optimizer, T_0=t0, T_mult=2, eta_min=lr_min)
        #scheduler=LinearWarmupCosineAnnealingLR(optimizer, 1, 100, warmup_start_lr=0.0)
        

        
        return optimizer
    
    def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        #print(self.ode.par.grad)
        #print(self.ode.p.grad)
        for p in self.contextEnc.parameters():
            print(p.grad)
        print('these are the grads for meta model')    
        for p in self.ode.nn.parameters():
            print(p.grad)
        
        #norms = grad_norm(self.contextEnc, norm_type=2)
        
        
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
        TMP = TMP_1[:, :, 0:self.dimD]
        return TMP 
    
    
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
        
        #self.par = out   
       
        #self.ode.par = (torch.ones((self.batch_size))*0.1).to(device)
        self.ode.par = out
        
        TMP = self.generateTMP(UV,self.ode.par)
        print(self.ode.t)
        self.ode.t = 0
        #TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = TMP[:,:,0:self.dimD]
        #TMP = TMP.permute(1,0,2)
        if self.isSurfaceTMP:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        #print(a,par)
        param_loss = F.mse_loss(self.ode.par, a)
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
        

        #TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = self.generateTMP(UV,self.ode.par)
        #TMP = TMP[:,:,0:self.dimD]
        #TMP = TMP.permute(1,0,2)
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
    

    
    
class OnlyGCN(L.LightningModule):
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
        
        self.ode = APModel2(self.S,self.par, self.batch_size,self.dimD,edges,pseudo,degree,norm,bias,root_weight)
        self.contextEnc =  ContextEncoderLSTM()
        self.surfaceIds = sufaceIds
        self.isPhysics = isPhysics
        self.isNeural = isNeural
        self.isRecon = isRecon
        self.isSurfaceTMP = surfaceTMP
        self.save_hyperparameters()
        
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        
        optimizer = torch.optim.Adam(self.parameters(),lr=1e-2,weight_decay=0.0001) 
        #optimizer = torch.optim.Adam([{'params':self.ode.parameters(),'weight_decay':0.00001, 'lr':1e-2},
        #                              {'params':self.contextEnc.parameters(),'weight_decay':0.0001, 'lr':1e-3}])
        #optimizer.add_param_group({'params':self.metaNet.parameters(),'weight_decay':0.1, 'lr':1e-5})
        #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #optimizer, T_0=t0, T_mult=2, eta_min=lr_min)
        #scheduler=LinearWarmupCosineAnnealingLR(optimizer, 1, 100, warmup_start_lr=0.0)
        

        
        return optimizer
    
    """def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        #print(self.ode.par.grad)
        #print(self.ode.p.grad)
        for p in self.contextEnc.parameters():
            print(p.grad)
        print('these are the grads for meta model')    
        for p in self.ode.nn.parameters():
            print(p.grad)
        
        #norms = grad_norm(self.contextEnc, norm_type=2)"""
        
        
    def generateTMP(self,UV):
        t0 = torch.tensor(0.0)
                
        for i,t in enumerate(self.t[1:]):
            #print(i)        
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
                #p = par.unsqueeze(1)
                #UV=torch.cat((UV,p),dim=1)
                TMP = odesolve_adjoint(self.ode, UV, options = options)
                UV_temp=UV.unsqueeze(1)
                TMP_1=TMP.unsqueeze(1)
                TMP_1 = torch.cat((UV_temp,TMP_1), 1)
            else:
                TMP = TMP[:,:self.dimD]
                #p = par.unsqueeze(1)
                #TMP=torch.cat((TMP,p),dim=1)
                TMP = odesolve_adjoint(self.ode, TMP, options = options)
                TMP_new=TMP.unsqueeze(1)
                TMP_1=torch.cat((TMP_1,TMP_new),1)
            t0 = t
        TMP = TMP_1[:, :, 0:self.dimD]
        return TMP 
    
    
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
        #k=k.view(-1,seq_len,dim)
        #out=self.contextEnc(k)
        #out=out.view(bs,-1)
        #out=torch.mean(out,dim=1)
        
        #print(par)
        #par=par.squeeze()
        
        #self.par = out   
       
        #self.ode.par = (torch.ones((self.batch_size))*0.1).to(device)
        #self.ode.par = out
        
        #TMP = self.generateTMP(UV[:,0:self.dimD])
        
        self.ode.t = 0
        TMP=odeint(self.ode, UV[:,0:self.dimD], self.t, method='dopri5')
        print(self.ode.t)
        TMP = TMP[:,:,0:self.dimD]
        TMP = TMP.permute(1,0,2)
        if self.isSurfaceTMP:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        #print(a,par)
        #param_loss = F.mse_loss(self.ode.par, a)
        loss = F.mse_loss(TMP, y) 
        self.log_dict({"train": loss},prog_bar=True,on_step=True)
        #lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print(lr)
        
        
        
        
        return loss #+ 100*param_loss
    
    def validation_step(self, train_batch, train_idx) -> STEP_OUTPUT:
        
        
        UV, y, k, a = train_batch
        
        
        #k =k.squeeze(1)
        bs, num_k, seq_len, dim=k.size()
        #k = nn.Parameter(k, requires_grad=True)
        #par=torch.nn.Parameter(torch.zeros(bs,1).to(device))
        
       
        
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
        
        #self.ode.par = out
        #TMP=odeint(self.ode, UV, self.t, method='rk4',options={'step_size':0.01})
        #self.ode.p = k
        

        #TMP=odeint(self.ode, UV, self.t, method='dopri5')
        TMP = self.generateTMP(UV[:,:self.dimD])
        #TMP = TMP[:,:,0:self.dimD]
        #TMP = TMP.permute(1,0,2)
        if self.isSurfaceTMP:
            TMP = TMP.permute(0,2,1)
            TMP = TMP[:,self.surfaceIds,:].squeeze().permute(0,2,1)
        
        #print(a,par)
        #param_loss = F.mse_loss(a, out)
        
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
