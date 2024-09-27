import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph


def read_matrix(file_name):
    return np.fromfile(file_name)

device = 'cuda:0'
full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
full_heart=full_heart.reshape((-1,3))
    
a=kneighbors_graph(full_heart, 6)
arr=a.toarray()

points=np.where(arr==1)
edges=np.reshape(points, (2,-1))
   

dist = list(map(lambda x, y: np.linalg.norm(x-y), full_heart[edges[0]], full_heart[edges[1]]))
normDist = dist/np.linalg.norm(dist) 
        


        
edges=torch.from_numpy(edges).to(device)
    
pseudo = torch.from_numpy(normDist).to(torch.float32).to(device)