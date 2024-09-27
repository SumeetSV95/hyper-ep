
# First networkx library is imported  
# along with matplotlib 
import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_spline_conv import spline_conv
import torch
from dataset import TMPDataset
from torch.utils.data import DataLoader



   
device = "cuda" if torch.cuda.is_available() else "cpu"
# Defining a Class 
def read_matrix(file_name):
    return np.fromfile(file_name)
full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
full_heart=full_heart.reshape((-1,3))
print(full_heart.shape)
a=kneighbors_graph(full_heart, 3)
arr=a.toarray()

points=np.where(arr==1)
edges=np.reshape(points, (2,-1))
print(points)
print(full_heart[edges[0]])
print(full_heart[edges[1]])

dist1 = np.abs(full_heart[edges[0]][:,0] - full_heart[edges[1]][:,0])
dist2 = np.abs(full_heart[edges[0]][:,1] - full_heart[edges[1]][:,1])
dist3 = np.abs(full_heart[edges[0]][:,2] - full_heart[edges[1]][:,2])
normDist1 = dist1/np.linalg.norm(dist1)
normDist2 = dist2/np.linalg.norm(dist2)
normDist3 = dist3/np.linalg.norm(dist3)


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(full_heart[:,0],full_heart[:,1],full_heart[:,2])
for i, j in zip(points[0], points[1]):
    ax.plot([full_heart[i][0], full_heart[j][0]],[full_heart[i][1],full_heart[j][1]], zs=[full_heart[i][2],full_heart[j][2]],c='blue')   
plt.savefig('graph.png')
dataset=TMPDataset("/home/sv6234/ECGI/data/TMP_data_UV_new","/home/sv6234/ECGI/data/TMP_data_GT_new")
train_set, valid_set, test_set=torch.utils.data.random_split(dataset,[1188,297,371])

train_loader=DataLoader(train_set,batch_size=16,shuffle=True)
uv,label=next(iter(train_loader))
print(uv.shape, label.shape)
dimD = 1862
u = torch.t(uv[:,0:dimD])
            
v = torch.t(uv[:,dimD:dimD*2])
features=torch.cat((u,v),1)
print(features.shape)
#node_features=torch.rand((1862,32),requires_grad=True,dtype=torch.float32).to(device)
#print(node_features.shape)
edges=torch.from_numpy(edges).to(device)
print(edges.shape)
pseudo1 = torch.from_numpy(normDist1).to(device).unsqueeze(0)
pseudo2 = torch.from_numpy(normDist2).to(device).unsqueeze(0)
pseudo3 = torch.from_numpy(normDist3).to(device).unsqueeze(0)
print(pseudo1.shape)
pseudo=torch.cat((pseudo1,pseudo2,pseudo3),0).permute(1,0).to(torch.float32)
print(pseudo.shape)

weight = torch.rand((25,32,16),dtype=torch.float32).to(device).to(device)
kernel_size = torch.tensor([5, 5, 5]).to(device) 


is_open_spline = torch.tensor([1, 1, 1], dtype=torch.uint8).to(device)  # only use open B-splines
degree = 1  # B-spline degree of 1
norm = True 

"""out = spline_conv(features,edges,pseudo,weight,kernel_size,is_open_spline,degree, norm,bias=None, root_weight=None)
print(out)
print(out.shape)"""


