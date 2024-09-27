import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader, random_split, Sampler
import random
from torch.utils.data import Sampler
import random
import math

device = 'cuda:0'
class TMPDataset(Dataset):
    def __init__(self, data_dir, labels_dir, k_shot,H=None,transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.data_idx, self.kshot_idx = self.split()
        self.k_shot = k_shot
        self.H = H
    
    def __len__(self):
        
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        #use fixed data not random. generate random numbers for all the data points and then use them for the id. Generate a dictionary of ids for all samples. 
        index,val = self.data_idx[idx]
        
        a = None
        if val == 1:
            data_dir = os.path.join(self.data_dir,'0.08')
            label_dir = os.path.join(self.labels_dir,'0.08')
            a = 0.08
        elif val ==2:
            data_dir = os.path.join(self.data_dir,'0.1')
            label_dir = os.path.join(self.labels_dir,'0.1')
            a = 0.1
        elif val==3:
            data_dir = os.path.join(self.data_dir,'0.12')
            label_dir = os.path.join(self.labels_dir,'0.12')
            a = 0.12
        elif val==4:
            data_dir = os.path.join(self.data_dir,'0.14')
            label_dir = os.path.join(self.labels_dir,'0.14')
            a = 0.14
        elif val ==5:
            data_dir = os.path.join(self.data_dir,'0.16')
            label_dir = os.path.join(self.labels_dir,'0.16')
            a = 0.16
        elif val ==6:
            data_dir = os.path.join(self.data_dir,'0.09')
            label_dir = os.path.join(self.labels_dir,'0.09')
            a=0.09
        elif val ==7:
            data_dir = os.path.join(self.data_dir,'0.11')
            label_dir = os.path.join(self.labels_dir,'0.11')
            a=0.11    
        elif val ==8:
            data_dir = os.path.join(self.data_dir,'0.13')
            label_dir = os.path.join(self.labels_dir,'0.13')
            a=0.13 
               
        data_path = os.path.join(data_dir,str(index)+".pt")
        label_path = os.path.join(label_dir, str(index)+".pt")
        data = torch.load(data_path).to(device)
        label = torch.load(label_path).to(device)
        #arr=np.random.randint(0,len(self.kshot_idx)-1,self.k_shot,replace=False)
        arr = random.sample(range(0,len(self.kshot_idx)-1),self.k_shot)
        for j ,i in enumerate(arr):
            if j == 0:
                id=self.kshot_idx[i]
                label_path = os.path.join(label_dir, str(id)+".pt")
                k_label = torch.load(label_path).to(device).unsqueeze(0)
                if self.H is not None:
                    k_label = torch.matmul(self.H, k_label.permute(0,2,1)).permute(0,2,1)
                
            else:
                id=self.kshot_idx[i]
                label_path = os.path.join(label_dir, str(id)+".pt")
                t=torch.load(label_path).to(device).unsqueeze(0)
                if self.H is not None:
                    temp = torch.matmul(self.H, t.permute(0,2,1)).permute(0,2,1)
                    k_label = torch.cat((k_label, temp),dim=0)
                else:
                    k_label = torch.cat((k_label, t),dim=0)
                
        return (data, label, k_label, torch.tensor(a))
    
    def split(self):
        
        lis = []
        
        for i in range(0,1856):
        #for i in range(0,64):
            lis.append(i)
        k_shot=np.random.choice(lis,96,replace=False)
        #k_shot=np.random.choice(lis,20,replace=False)
        
        data_idx = []
        for i in range(1856):
        #for i in range(0,64):
            if i not in k_shot:
                val=np.random.randint(1,5)
                #val=np.random.randint(2,3)
                data_idx.append((i,val))
        
        return data_idx,k_shot
    #def reset(self, indices):
    
class TMPDatasetEQ(Dataset):
    def __init__(self, data_dir, labels_dir, k_shot, H=None, transform=None) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.k_shot = k_shot
        self.H = H
        self.data_idx, self.kshot_idx = self.split()
        

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        
        
        if idx >= len(self.data_idx):
            raise IndexError(f"Index {idx} is out of range for data_idx with length {len(self.data_idx)}.")
        
        index, val = self.data_idx[idx]
        
        
        # Handle different val values
        if val == 1:
            data_dir = os.path.join(self.data_dir, '0.08')
            label_dir = os.path.join(self.labels_dir, '0.08')
            a = 0.08
        elif val == 2:
            data_dir = os.path.join(self.data_dir, '0.1')
            label_dir = os.path.join(self.labels_dir, '0.1')
            a = 0.1
        elif val == 3:
            data_dir = os.path.join(self.data_dir, '0.12')
            label_dir = os.path.join(self.labels_dir, '0.12')
            a = 0.12
        elif val == 4:
            data_dir = os.path.join(self.data_dir, '0.14')
            label_dir = os.path.join(self.labels_dir, '0.14')
            a = 0.14
        else:
            raise ValueError(f"Unexpected class value: {val}")
        
        data_path = os.path.join(data_dir, str(index) + ".pt")
        label_path = os.path.join(label_dir, str(index) + ".pt")
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            raise FileNotFoundError(f"Data or label file does not exist for index {index} and class {val}.")
        
        data = torch.load(data_path).to(device)
        label = torch.load(label_path).to(device)
        
        arr = random.sample(range(0, len(self.kshot_idx) - 1), self.k_shot)
        
        for j, i in enumerate(arr):
            
            if j == 0:
                id = self.kshot_idx[i]
                label_path = os.path.join(label_dir, str(id) + ".pt")
                k_label = torch.load(label_path).to(device).unsqueeze(0)
                if self.H is not None:
                    k_label = torch.matmul(self.H, k_label.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                id = self.kshot_idx[i]
                label_path = os.path.join(label_dir, str(id) + ".pt")
                t = torch.load(label_path).to(device).unsqueeze(0)
                if self.H is not None:
                    temp = torch.matmul(self.H, t.permute(0, 2, 1)).permute(0, 2, 1)
                    k_label = torch.cat((k_label, temp), dim=0)
                else:
                    k_label = torch.cat((k_label, t), dim=0)
        
        return (data, label, k_label, torch.tensor(a),idx)
    
    def split(self):
        lis = list(range(1856))  # Assuming 1856 is the total number of data points
        k_shot = np.random.choice(lis, 96, replace=False)
        
        data_idx = []
        num_classes = 4
        samples_per_class = (1856 - len(k_shot)) // num_classes
        
        for val in range(1, num_classes + 1):
            class_indices = [i for i in lis if i not in k_shot]
            if len(class_indices) < samples_per_class:
                raise ValueError(f"Not enough data points to assign {samples_per_class} samples to class {val}.")
            selected_indices = random.sample(class_indices, samples_per_class)
            data_idx.extend([(i, val) for i in selected_indices])
            lis = [i for i in lis if i not in selected_indices]
        
        return data_idx, k_shot
    


class BalancedBatchSampler(Sampler):
    def __init__(self, data_idx, train_indices, batch_size, num_classes):
        self.data_idx = [data_idx[i] for i in train_indices]
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = batch_size // num_classes
        
        # Group indices by class
        self.class_indices = {i: [] for i in range(1, num_classes + 1)}
        for idx, (data_idx_value, val) in enumerate(self.data_idx):
            self.class_indices[val].append(idx)  # Store the index in data_idx
        
        self.total_batches = min(len(v) // self.samples_per_class for v in self.class_indices.values())
    
    def __iter__(self):
        batch_count = 0
        class_indices_copy = {k: v.copy() for k, v in self.class_indices.items()}
        
        while batch_count < self.total_batches:
            batch = []
            for class_idx in range(1, self.num_classes + 1):
                if len(class_indices_copy[class_idx]) < self.samples_per_class:
                    class_indices_copy[class_idx] = self.class_indices[class_idx].copy()

                selected_indices = random.sample(class_indices_copy[class_idx], self.samples_per_class)
                for idx in selected_indices:
                    class_indices_copy[class_idx].remove(idx)
                batch.extend(selected_indices)

            if len(batch) == self.batch_size:
                
                random.shuffle(batch)
                yield batch
                batch_count += 1
            else:
                break
    
    def __len__(self):
        return self.total_batches





        
        
            
if __name__=="__main__":
    """    def read_matrix(file_name):
        return np.fromfile(file_name)
    H=read_matrix('/home/sv6234/ECGI/data/EC/Trans.bin')
    H=torch.from_numpy(H.reshape(396,1862)).to(torch.float32).to(device)
    dataset=TMPDataset("/home/sv6234/ECGI/data/TMP_data_UV_new/","/home/sv6234/ECGI/data/TMP_data_GT_new/",10,H)
    train_set, test_set=torch.utils.data.random_split(dataset,[1408,352])
    train_loader=DataLoader(train_set,batch_size=8,shuffle=True,num_workers=1)
    test_loader=DataLoader(test_set,batch_size=8,shuffle=True)
    
    data=train_loader.dataset.__getitem__(0)
    data, label, k_label, a = data
    print(data.shape, label.shape, k_label.shape, a)
    print(len(train_loader.dataset.__dict__['indices']))"""
    
    def read_matrix(file_name):
        return np.fromfile(file_name)
    
    H = read_matrix('/home/sv6234/ECGI/data/EC/Trans.bin')
    H = torch.from_numpy(H.reshape(396, 1862)).to(torch.float32).to(device)
    dataset = TMPDatasetEQ("/home/sv6234/ECGI/data/TMP_data_UV_new/", "/home/sv6234/ECGI/data/TMP_data_GT_new/", 15, H)
    
    train_set, test_set = random_split(dataset, [1408, 352])
    
    # Using the custom sampler
    num_classes = 4
    batch_size = 16
    
    # Create a new sampler using the indices of the train_set
    train_indices = train_set.indices
    sampler = BalancedBatchSampler(dataset.data_idx, train_indices, batch_size, num_classes)
    
    train_loader = DataLoader(train_set, batch_sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    train_sampled_indices = set()
    test_sampled_indices = set()

    # Collect all indices sampled by the train_loader
    for batch in train_loader:
        indices = batch[-1]  # Assuming the indices are stored in the last element of the batch tuple
        train_sampled_indices.update(indices.cpu().numpy())

    # Collect all indices sampled by the test_loader
    for batch in test_loader:
        indices = batch[-1]  # Assuming the indices are stored in the last element of the batch tuple
        test_sampled_indices.update(indices.cpu().numpy())

    # Check for overlap between sampled indices
    overlap = train_sampled_indices.intersection(test_sampled_indices)

    if overlap:
        print(f"Warning: Overlap detected in {len(overlap)} indices between train and test loaders!")
    else:
        print("Success: No overlap detected between train and test loaders.")

    
    
    
    