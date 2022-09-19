import numpy as np
import torch
from torch import nn
import os
from numpy import *
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from Dynamic_output import decoding 


class Linear_train(nn.Module):
  def __init__(self,t_w,b_w):
    super(Linear_train, self).__init__()
    self.dropout = nn.Dropout(p=0.3)
    self.fc = nn.Linear(64,1)
    self.fc.weight=t_w
    self.fc.bias=b_w
#     with torch.no_grad():
#       self.fc.weight.copy_(t_w)
#       self.fc.bias.copy_(b_w)

  def forward(self, x):
    x = self.dropout(x)
    x = self.fc(x)
    x = torch.sigmoid(x)
    
    return x

def output_train(n_epochs,lr,t_w,b_w,x_train_input,y_train,x_test_input,y_test,device):
  linear_t = Linear_train(t_w,b_w).to(device)
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(linear_t.parameters(),lr=lr)
  val_accuracy_save=np.ones(n_epochs)
  acc_accuracy_save=np.ones(n_epochs)
  for epoch in range(n_epochs):
    linear_t.train()
    optimizer.zero_grad()
    y_out = linear_t(x_train_input)
    loss=criterion(y_out[:,-1,0], y_train)
    loss.backward(retain_graph=True)
    optimizer.step()
    with torch.no_grad():
      linear_t.fc.bias=b_w
      # linear_t.fc.weight=t_w
    y_a_t = linear_t(x_train_input)
    accuracy=decoding.get_accuracy(y_train,y_a_t[:,-1,0])
    y_v_t = linear_t(x_test_input)
    val_accuracy=decoding.get_accuracy(y_test,y_v_t[:,-1,0])
    loss_print=loss
    accuracy_print=accuracy
    print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss_print},accuracy={accuracy_print},val_accuracy={val_accuracy}")
    val_accuracy_save[epoch]=val_accuracy
    acc_accuracy_save[epoch]=accuracy_print

  return linear_t.fc.weight,linear_t.fc.bias,accuracy,val_accuracy,y_a_t,y_v_t,val_accuracy_save,acc_accuracy_save


def run_dynamic_output(min_t,max_t,data_behavior,data_control,num_fold,lr,epochs,batch_size,n_epochs,out_lr,validation,using_all,verbose,device):
  X_t,y_t=decoding.data_processing(data_behavior,data_control)
  X_t=X_t[:,min_t:max_t,:]
  acc_save=[]
  val_save=[]
  acc_ori=[]
  val_ori=[]
  w_save=[]
  b_save=[]
  model_all=[]
  x_save_all=[]
  x_save_val_all=[]
  y_a_all=[]
  y_v_all=[]
  y_a_check=[]
  y_v_check=[]
  t_w_all=[]
  b_w_all=[]
  val_visual=[]
  acc_visual=[]
  kf = KFold(n_splits=num_fold,random_state=42, shuffle=True) # apply k_fold to compute the average of accuracy lose
  vi_s=0
  for train_index, test_index in kf.split(X_t):
    print('start:',vi_s)
    vi_s=vi_s+1
    X_t_train, X_t_test = X_t[train_index], X_t[test_index] 
    y_t_train, y_t_test = y_t[train_index], y_t[test_index]

    X_train = torch.tensor(X_t_train,device=device).float()
    y_train = torch.tensor(y_t_train,device=device).float()

    X_test=torch.tensor(X_t_test,device=device).float()
    y_test=torch.tensor(y_t_test,device=device).float()
    
    model,acc,val_acc,x_save,x_save_val,y_a,y_v=decoding.rnn_model(lr,epochs,batch_size,X_train,y_train,X_test,y_test,
                                         validation=validation,
                                         using_all=using_all,
                                         verbose=verbose,
                                         device=device)
    acc_ori.append(acc)
    val_ori.append(val_acc)
    y_a_all.append(y_a.cpu().detach())
    y_v_all.append(y_v.cpu().detach())
    model_all.append(model)
    t_w=model.fc.weight
    b_w=model.fc.bias
    t_w_all.append(t_w.data.cpu().numpy().copy())
    b_w_all.append(b_w.data.cpu().numpy().copy())
    acc_all=np.ones(x_save.shape[1])
    val_all=np.ones(x_save.shape[1])
    w_fold=[]
    b_fold=[]
    y_a_fold=[]
    y_v_fold=[]
    val_accuray_save_fold=[]
    acc_accuray_save_fold=[]
    for i in range(x_save.shape[1]):
      x_train_input=x_save[:,i:i+1,:]
      x_test_input=x_save_val[:,i:i+1,:]
      linear_weight,linear_bias,acc,val,y_a_t,y_v_t,val_accuracy_save,acc_accuracy_save=output_train(n_epochs,out_lr,t_w,b_w,x_train_input,y_train,x_test_input,y_test,device)
      acc_all[i]=acc
      val_all[i]=val
      w_fold.append(linear_weight.cpu().detach())
      b_fold.append(linear_bias.cpu().detach())
      y_a_fold.append(y_a_t.cpu().detach())
      y_v_fold.append(y_v_t.cpu().detach())
      val_accuracy_save_fold.append(val_accuracy_save)
      acc_accuracy_save_fold.append(acc_accuracy_save)
    w_save.append(np.array(w_fold))
    b_save.append(np.array(b_fold))
    acc_save.append(acc_all)
    val_save.append(val_all)
    val_visual.append(val_accuracy_save_plot)
    acc_visual.append(acc_accuracy_save_plot)
    x_save_all.append(x_save.cpu().detach())
    x_save_val_all.append(x_save_val.cpu().detach())
    y_a_check.append(np.array(y_a_fold))
    y_v_check.append(np.array(y_v_fold))
    
  return np.array(acc_save),np.array(val_save),np.array(w_save),np.array(b_save),model_all,np.array(x_save_all),np.array(x_save_val_all),np.array(acc_ori),np.array(val_ori),np.array(y_a_all),np.array(y_v_all),np.array(y_a_check),np.array(y_v_check),np.array(t_w_all),np.array(b_w_all),np.array(val_visual)

