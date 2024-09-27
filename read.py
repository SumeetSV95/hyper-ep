import codecs
from collections import Counter
from sklearn.neighbors import kneighbors_graph

    
import numpy as np

 # little-endian single precision float
def read_matrix(file_name):
    return np.fromfile(file_name)

surface = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Ref/heart_sur.cor")
surface = surface.reshape((-1,3))
print(surface.shape)
full_heart = read_matrix("/home/sv6234/ECGI/data/EC/geometry/Final/1862/heart.cor")
full_heart=full_heart.reshape((-1,3))

  #sample data
a=kneighbors_graph(full_heart, 6)
print(a)
graph=a.toarray()
print(graph.shape)

    