from torch.utils.data import DataLoader
from model import HybridModel
import torch
from dataset import TMPDataset
from sklearn.neighbors import kneighbors_graph
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import numpy as np
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.font_manager as font_manager
import scipy.io

class APModel(nn.Module):
    def __init__(self, S, par, batch_size, dimD) -> None:
        super(APModel, self).__init__()
        self.S = S
        self.par = par
        self.batch_size = batch_size
        self.dimD = dimD
    def forward(self,t,y):
        
        u = torch.t(y[:,0:self.dimD])
        
        v = torch.t(y[:,self.dimD:self.dimD*2])
       
        k = 8
        e = 0.01
        MK = torch.matmul(self.S, u) 
        
        pde1 = MK + k*u*(1-u)*(u-self.par) 
        pde2 = -e*(k*u*(u-self.par-1)+v)
        #print(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0).shape)
        return torch.t(torch.cat((pde1.squeeze(), pde2.squeeze()), dim=0))
def compute_gradient(tensor):
    # Compute the gradient along the last dimension
    gradient = tensor[..., 1:] - tensor[..., :-1]
    return gradient    
def get_activation_len_new(tmp):
    bs, time, features = tmp.shape
    start = torch.ones((bs, features), dtype=torch.int64)
    end = torch.full((bs, features), 375, dtype=torch.int64)
    act_len = torch.ones((bs, features), dtype=torch.int64)
    
    # Find the index of the maximum value for each (batch, feature) pair
    _, max_indices = torch.max(tmp, dim=1)
    
    for b in range(bs):
        for f in range(features):
            tm = tmp[b, :, f]
            index = max_indices[b, f].item()

            # Find the start index
            s = (tm[:index] >= 0.5).nonzero(as_tuple=True)[0]
            if len(s) > 0:
                start[b, f] = s[0].item()
            
            # Find the end index
            e = (tm[index:] <= 0.5).nonzero(as_tuple=True)[0]
            if len(e) > 0:
                end[b, f] = e[0].item() + index
            
            act_len[b, f] = end[b, f] - start[b, f]
    
    return act_len, start, end

def get_activation_len_vectorized(tmp):
    bs, time, features = tmp.shape
    active = (tmp > 0).float()
    
    # Finding changes to detect activation start and end
    starts = torch.cat((active[:, :1], active[:, 1:] - active[:, :-1]), dim=1) > 0
    ends = torch.cat((active[:, 1:] - active[:, :-1], -active[:, -1:]), dim=1) < 0

    # Prepare to calculate lengths and track starts and ends
    lengths = torch.zeros_like(active)
    start_tracker = torch.full((bs, features), -1)  # -1 will indicate no activation found
    end_tracker = torch.full((bs, features), -1)

    # Compute lengths where activations are valid
    for b in range(bs):
        for f in range(features):
            start_pos = starts[b, :, f].nonzero(as_tuple=False).squeeze(-1)
            end_pos = ends[b, :, f].nonzero(as_tuple=False).squeeze(-1)

            for s, e in zip(start_pos, end_pos):
                length = e - s + 1
                if length > 35:
                    lengths[b, s:e+1, f] = length  # Assign length to all positions in the activation
                    # Track the start and end of the maximum activation length
                    current_max_length = (lengths[b, :, f] == lengths[b, :, f].max()).float() * length
                    max_length_mask = current_max_length == lengths[b, :, f].max()
                    start_tracker[b, f] = torch.where(max_length_mask, s, start_tracker[b, f]).max()
                    end_tracker[b, f] = torch.where(max_length_mask, e, end_tracker[b, f]).max()

    # Max length for each feature across all time
    max_lengths = torch.max(lengths, dim=1)[0]
    start_tracker = start_tracker.float()
    end_tracker = end_tracker.float()
    return max_lengths, start_tracker, end_tracker



def get_activation_len(tmp):
    start = None
    end = None
    #print(np.round(tmp,4))
    for i in range(0,375):
        
        if start is None and round(tmp[i],4)>0:
            start = i
        if start is not None and end is None and round(tmp[i],4)<=0:
            end = i
            if end - start <=35:
                start = None 
                end = None
    #print(start, end)
    if start is not None and end is not None:
        return end-start
    
    else:
        return float('inf')
def read_matrix(file_name):
    return np.fromfile(file_name)


dimD = 1862
batch_size = 16

device='cuda:0'
par = torch.from_numpy(np.full((1862,batch_size),0.15)).to(torch.float32).to(device)
full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
full_heart=full_heart.reshape((-1,3))
    
 
par = torch.from_numpy(np.full((1862,1),0.15)).to(torch.float32).to(device)
"""a=read_matrix('/home/sv6234/ECGI/data/EC/Trans_state.bin')
S=torch.from_numpy(a.reshape((1862,1862))).to(torch.float32).to(device)"""
mat = scipy.io.loadmat('/home/sv6234/ECGI/S_Trans.mat')
mat = mat['S']
S=torch.from_numpy(mat).float().to(device)
 

#dataset=TMPDataset("/home/sv6234/ECGI/data/TMP_data_UV_new/1.5","/home/sv6234/ECGI/data/TMP_data_GT_new/1.5")
#142
#model = HybridModel1.load_from_checkpoint("/home/sv6234/ECGI/checkpoints/best-model-epoch=03-val_loss=0.02.ckpt")
#model2 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkpoints2/best-model-epoch=09-val_loss=0.04.ckpt").to(device)
#model2 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkHybridSurface_full/best-model-epoch=15-val_loss=0.041.ckpt").to(device)
#dataset=TMPDataset("/home/sv6234/ECGI/data/TMP_data_UV_new/","/home/sv6234/ECGI/data/TMP_data_GT_new/",10)
model3 = HybridModel.load_from_checkpoint('/home/sv6234/ECGI/checkHybrid_TMP_Full_seed_1250/best-model-epoch=09-val_loss=0.011.ckpt').to(device)
#train_set, valid_set, test_set=torch.utils.data.random_split(dataset,[44,0,0])
#train_set, valid_set, test_set=torch.utils.data.random_split(dataset,[1056,352,352])
#train_loader=DataLoader(train_set,batch_size=batch_size,shuffle=True)



def generate_mat(tmp,output_path):
    #filePath = '/home/sv6234/ECGI/data/TMP_data_GT_new/0.pt'
    #tmp=torch.load(filePath)
    #tmp=tmp.numpy()
    print(tmp.shape)
    scipy.io.savemat(output_path, mdict={'U': tmp})

train_loader=torch.load('/home/sv6234/ECGI/test_loader_checkHybrid_TMP_full.pt')   
train_loader.shuffle = False

label_dict = {}
mse=torch.tensor(0.0).to(device)
mse1=torch.tensor(0.0).to(device)
uv = None
for it in range(1):    
    uv, label, k, a = train_batch=next(iter(train_loader))
    uv, label, k, a = uv[:batch_size], label[:batch_size], k[:batch_size], a[:batch_size]
    #act_len_label, start, end=get_activation_len_new(label)
    #print(act_len_label)
    #print(act_len_label.shape)
    #print(start, end, start.shape)
    a=a.to(device)
    p=torch.zeros((1862,batch_size)).to(device)
            #print(par)
    for i in range(batch_size):
        temp     = torch.full((1862,1),a[i].item())
        p[:,i]  = temp.squeeze()
    
    model1=APModel(S, p,1,dimD)
    t=torch.from_numpy(np.zeros(375)).to(torch.float64).to(device)
    val=0
    for i in range(375):
            t[i] = val
            val += 0.9
        
    

    #print(TMP.shape)
    
    """tmp=model(uv)
    tmp = tmp.to('cpu').detach().numpy()"""
    k_prime = k[:,0:1,:,:]
    k_prime = k_prime.squeeze()
    # for recon model
    #tmp1=model2(uv,k,k_prime)
    #tmp1=model2(uv,k)
    tmp1 = torch.ones((batch_size,375,1862))
    print("this is k")
    print(k.shape)
    print(k_prime.shape)
    tmp2=model3(uv,k)
    print(tmp2.shape)
    #act_len_model, start_model, end_model=get_activation_len_new(tmp2)
    #print(act_len_model)
    #print(start_model, end_model, start.shape)
    #print(tmp2)

    #print(a,model2.ode.par)
    #print(tmp1, label)
    #mse += F.mse_loss(tmp1, label)
    #mse1 += F.mse_loss(TMP, label)
    param_mse = F.mse_loss(a,model3.ode.par.to(device))
    print(a)
    print(model3.ode.par)
    print("this is std")
    print('std of the model')
    #print(torch.std(label-tmp1))
    #print(torch.std(label-TMP))
    print('param mse')
    print(param_mse)
    print('parma std')
    print('act_error')
    #print(F.mse_loss(act_len_model,act_len_label))
    #print(F.mse_loss(start_model,start))
    #print(F.mse_loss(end_model,end))
    print(torch.std(a-model3.ode.par.to(device)))
    tmp1 = tmp1.to('cpu').detach().numpy()
    #TMP = TMP.to('cpu').detach().numpy()
    tmp2 = tmp2.to('cpu').detach().numpy()
    label=label.to('cpu').numpy()
    for i in range(batch_size):
        if a[i].item() not in label_dict:
            label_dict[a[i].item()] = [model3.ode.par[i].item()]
        else:
            label_dict[a[i].item()].append(model3.ode.par[i].item())
    #print(tmp.shape)
    print(label_dict)
    print(label.shape)

for key in label_dict:
    arr=np.array(label_dict[key])
    mean = np.mean(arr)
    std = np.std(arr)
    print(key, mean, std)
print('moddel mse:')
print(mse/1)
print(mse1/1)

btc=np.random.randint(0,batch_size,5)
par = torch.zeros(5).to(device)
for i,b in enumerate(btc):
    if i ==0:
        uv1=uv[b].unsqueeze(0)
        
    else:
        temp=uv[b].unsqueeze(0)
        
        
        uv1 = torch.cat((uv1,temp),dim=0)
        
        
        print(uv1.shape)
    par[i] =model3.ode.par[b]
model1.par = par
y=odeint(model1, uv1, t, method='dopri5') 
y=y.to('cpu')
TMP = y[:, :, 0:dimD] 
TMP=TMP.permute(1,0,2).to(device)
TMP = TMP.to('cpu').detach().numpy()
#print(TMP.shape)

for i,b in enumerate(btc):
    
    #(model1.par[0][b])
    #print(model2.ode.par[b])
    #print(model3.ode.par.shape)
    arr=np.random.randint(0,477,3)
    """model1.par = model3.ode.par[b]
    print(uv[b].unsqueeze(0).shape)
    y=odeint(model1, uv[b].unsqueeze(0), t, method='dopri5') 
    y=y.to('cpu')
    TMP = y[:, :, 0:dimD] 
    TMP=TMP.permute(1,0,2).to(device)
    TMP = TMP.to('cpu').detach().numpy()
    print(TMP.shape)"""
    for val in arr:
        
        #act=get_activation_len(label[b,:,val])
        #print('this is act: '+str(act))    
        #print(b,val)
        plt.rc('font',size=15)
        #act = act_len_label[b][val]
        #plt.plot(tmp[b,:,val],label='GCN')
        plt.plot(label[b,:,val],label='GT a ='+str(round(p[0][b].item(),4))  )
        #print(label[b,:,val],start[b][val].item())
        ar=label[b,:,val]
        #idx = int(start[b][val].item())
        #idx_end = int(end[b][val].item())
        #print(ar[idx])
        
        #act=get_activation_len(TMP[i,:,val])
        plt.plot(TMP[i,:,val],label='partial physics  a='+ str(round(p[0][b].item(),4)))
        #plt.plot(tmp1[b,:,val],label='Hybrid recon a ='+str(round(model2.ode.par[b].item(),4)) + " "+ str(F.mse_loss(torch.from_numpy(tmp1[b,:,val]),torch.from_numpy(label[b,:,val]))))
        #act=get_activation_len(tmp2[b,:,val])
        #act = act_len_model[b][val]
        plt.plot(tmp2[b,:,val],label='Hybrid a ='+str(round(model3.ode.par[b].item(),4)))
        """plt.scatter(start_model[b][val].item(),tmp2[b,:,val][int(start_model[b][val].item())].item())
        plt.scatter(end_model[b][val].item(),tmp2[b,:,val][int(end_model[b][val].item())].item())
        plt.scatter(start[b][val].item(),ar[idx])
        plt.scatter(end[b][val].item(),ar[idx_end])"""
        #plt.title(str(F.mse_loss( torch.from_numpy(label[b,:,val])))+" "+str(F.mse_loss(torch.from_numpy(tmp1[b,:,val]), torch.from_numpy(label[b,:,val])))+" "+ str(F.mse_loss(torch.from_numpy(TMP[b,:,val]), torch.from_numpy(label[b,:,val]))))
        font = font_manager.FontProperties( 
                                   weight='bold',
                                   style='normal', size=10)
        plt.legend(prop=font,loc='lower left')
        
        plt.savefig('results/presentation/ckpt_par_'+str(p[0][b])+"_"+str(val)+'.png')
        
        plt.close()
#plt.plot(tmp[b,:,:],linewidth=0.1)
#plt.savefig('batch.png')
#plt.close()
plt.plot(tmp2[b,:,:],linewidth=0.1)
plt.savefig('batch1.png')
plt.close()
for i in range(4):
    #generate_mat(tmp1[i,:,:], 'model_out_sur'+str(i)+'.mat')
    generate_mat(tmp2[i,:,:], 'model_out'+str(i)+'.mat')
    generate_mat(label[i,:,:], 'gt_out'+str(i)+'.mat')
plt.plot(label[b,:,:],linewidth=0.1)
plt.savefig('batch2.png')
plt.close()

#print(F.mse_loss(torch.from_numpy(tmp), torch.from_numpy(label)))
print(F.mse_loss(torch.from_numpy(tmp1), torch.from_numpy(label)))
print(F.mse_loss(torch.from_numpy(TMP), torch.from_numpy(label)))
