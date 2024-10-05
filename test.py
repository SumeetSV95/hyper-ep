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
from lightning.pytorch import Trainer

dimD = 1119

batch_size =16
exp_num = '1'
device = 'cuda:0'
model_choise = 'nn'

model2 = HybridModel.load_from_checkpoint("/home/sv6234/ECGI/checkHybrid_sur_in_sur_out/best-model-epoch=17-val_loss=0.010.ckpt").to(device)
test_loader=torch.load('/shared/rc/hyper-ep/hyper-ep/test_loader_checkHybrid_TMP_full.pt')
trainer = Trainer(default_root_dir='',limit_train_batches=int(636/batch_size),limit_val_batches=15,limit_test_batches=13,max_epochs=3,accelerator='gpu', devices=1,
                        num_sanity_val_steps=0)
trainer.test(model2, dataloaders=test_loader)

# Extract the DataFrames from the model instance
main_table_df = model2.main_table_df
act_loss_df = model2.act_loss_df
param_loss_df = model2.param_loss_df

# Display the DataFrames
print("Main Table:")
print(main_table_df)

print("\nActivation Length Loss Table:")
print(act_loss_df)

print("\nParameter Loss Table:")
print(param_loss_df)