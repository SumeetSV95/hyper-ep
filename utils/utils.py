import scipy
import torch
import os

def generate_mat(tmp,output_path):
    scipy.io.savemat(output_path, mdict={'U': tmp})
    
def get_activation_len_new(tmp):
    bs, time, features = tmp.shape
    start = torch.zeros((bs, features), dtype=torch.float32)
    end = torch.zeros((bs, features), dtype=torch.float32)
    act_len = torch.zeros((bs, features), dtype=torch.float32)
    
    # Find the index of the maximum value for each (batch, feature) pair
    _, max_indices = torch.max(tmp, dim=1)
    
    for b in range(bs):
        for f in range(features):
            tm = tmp[b, :, f]
            index = max_indices[b, f].item()
            
            # Check if the value at the max index is greater than 0.5
            if tm[index] > 0.5:
                # Find the start index
                s = (tm[:index] >= 0.5).nonzero(as_tuple=True)[0]
                if len(s) > 0:
                    start[b, f] = s[0].item()
                
                # Find the end index
                e = (tm[index:] <= 0.5).nonzero(as_tuple=True)[0]
                if len(e) > 0:
                    end[b, f] = e[0].item() + index
                
                act_len[b, f] = end[b, f] - start[b, f]
            else:
                start[b, f] = 0
                end[b, f] = 0
                act_len[b, f] = 0
    
    return act_len, start, end

def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Folder already exists at: {path}")