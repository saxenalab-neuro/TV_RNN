import numpy as np
import torch
from torch import nn
import os
from numpy import *
import decoding 
import analysis
import training


data_behavior=np.load('./data/demo_session_behavior.npy',allow_pickle=True)
data_control=np.load('./data/demo_session_control.npy',allow_pickle=True)

torch.random.manual_seed(1) # for reproducing
np.random.seed(1)

########### hyper parameter setting for standard RNN ###########
lr=0.0001
epochs=20 # epochs for training RNN-S1
en_epochs=5 # epochs for training RNN-S2, the training is very slow, keep #epochs small if unnecessary
batch_size=5
min_t=600 # The overall #time points is 1800 for 60 secs, so 600-900 means -10s before the behavior to the behavior. 
max_t=900
########### hyper parameter setting for TV RNN ############
lr_tv=0.0001
epochs_tv=20
batch_size_tv=128
num_tv=10
num_fold=5
t_ind_ini=np.linspace(0,max_t-min_t,num_tv+1).astype(int)
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
###########################################
data_input,data_label=decoding.data_processing(data_behavior,data_control)
data_input=data_input[:,min_t:max_t,:]
data_input=torch.tensor(data_input,dtype=dtype,device=device)
data_label=torch.tensor(data_label,dtype=dtype,device=device)
input_s=data_input.shape[2]

save_val_S1,save_val_S2,save_val_tv=training.start_train(lr,
                                                epochs,
                                                en_epochs,
                                                batch_size,
                                                lr_tv,
                                                epochs_tv,
                                                batch_size_tv,
                                                num_tv,
                                                num_fold,
                                                data_input,
                                                data_label,
                                                t_ind_ini,
                                                s1=True,
                                                s2=True,
                                                tv=True,
                                                save_model=True,
                                                path='./save/',
                                                device=device)