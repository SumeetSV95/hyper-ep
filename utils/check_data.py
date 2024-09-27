import torch
from collections import defaultdict
import matplotlib.pyplot as plt
device = 'cuda:0'
train_loader = torch.load('/home/sv6234/ECGI/test_loader_checkHybrid_EGM_in.pt')

incorrect_samples=defaultdict(int)
for i,batch in enumerate(train_loader):
    print(i)
    UV, y, k, a = batch #y bs, 375, 1862
    y=y.to(device)
    bs,time,space = y.shape
    _, max_indices = torch.max(y, dim=1)
    for b in range(bs):
        for f in range(space):
            tm = y[b, :, f]
            index = max_indices[b, f].item()
            
            # Check if the value at the max index is greater than 0.5
            if tm[index] < 0.5:
                incorrect_samples[a[b].item()]+=1
                break
                
print(incorrect_samples)
                
    
