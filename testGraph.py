import networkx as nx 
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.neighbors import kneighbors_graph
from torch_spline_conv import spline_conv
import torch
from dataset import TMPDataset
from torch.utils.data import DataLoader

def read_matrix(file_name):
    return np.fromfile(file_name)
full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
full_heart=full_heart.reshape((-1,3))

with open('/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.idx', 'rb') as file:
    # Read data from the file using numpy.fromfile
    idx = np.fromfile(file, dtype=np.int32)
ids=np.where(idx>0)
print(ids)

surface = full_heart[ids]

a=kneighbors_graph(surface, 3)
arr=a.toarray()

points=np.where(arr==1)
edges=np.reshape(points, (2,-1))
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(surface[:,0],surface[:,1],surface[:,2])
ax.scatter3D(surface[0,0],surface[0,1],surface[0,2], c='r')
for i, j in zip(points[0], points[1]):
    ax.plot([surface[i][0], surface[j][0]],[surface[i][1],surface[j][1]], zs=[surface[i][2],surface[j][2]],c='blue') 
plt.savefig('graph.png')