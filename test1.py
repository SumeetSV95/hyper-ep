import os
import numpy as np
import torch
from torch_spline_conv import spline_conv


data_path = '/home/sv6234/pytorch_spline_conv_sumeet/data_coarsen/1-L/1-L_01242014-USPlaneNONE'
point_path = os.path.join(data_path, 'points.npy')
edge_path = os.path.join(data_path, 'edges.npy')

points = np.load(point_path)
edges = np.array(np.load(edge_path), dtype='int64')
edges = edges - 1 # matlab index from 1 but python from 0, so transform from matlab index to python index


pre_points = points[0]
libr_points = points[1]
gt_points = torch.tensor(points[2], dtype=torch.float, requires_grad=False)

x = np.concatenate([pre_points, libr_points], axis=1) # 2044 nodes with 6 (3+3) features each
x = torch.tensor(x, dtype=torch.float)

print(x.size())

edge_index = torch.tensor(edges)  # 18898 edges
print(edge_index.size())

pseudo = torch.rand((edge_index.size(1), 2), dtype=torch.float, requires_grad=True)  # two-dimensional edge attributes
weight = torch.rand((30, 6, 1024), dtype=torch.float, requires_grad=True)  # 30 parameters for in_channels x out_channels
kernel_size = torch.tensor([5, 5])  # 7 parameters in each edge dimension
is_open_spline = torch.tensor([1, 1], dtype=torch.uint8)  # only use open B-splines
degree = 1  # B-spline degree of 1
norm = True  # Normalize output by node degree.

root_weight = torch.rand((6, 1024), dtype=torch.float, requires_grad=True)  # separately weight root nodes
bias = None  # do not apply an additional bias

encode = spline_conv(x, edge_index, pseudo, weight, kernel_size,
                  is_open_spline, degree, norm, root_weight, bias)

pseudo1 = torch.rand((edge_index.size(1), 2), dtype=torch.float, requires_grad=True)  # two-dimensional edge attributes
weight1 = torch.rand((25, 1024, 3), dtype=torch.float, requires_grad=True)  # 25 parameters for in_channels x out_channels
root_weight1 = torch.rand((1024, 3), dtype=torch.float, requires_grad=True)  # separately weight root nodes

decode = spline_conv(encode, edge_index, pseudo1, weight1, kernel_size,
                  is_open_spline, degree, norm, root_weight1, bias)

print(decode.size())

loss = torch.nn.MSELoss()
output = loss(decode, gt_points)

output.backward()




