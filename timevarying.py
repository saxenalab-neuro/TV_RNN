import numpy as np
import sklearn
import torch
from torch import nn
import matplotlib.pyplot as plt
import random
import os
import torch.optim as optim
from torch.autograd import Variable
import decoding

class Model(nn.Module):
  def __init__(self,input_shape,num_tv,model_last):
    super(Model, self).__init__()
    self.input_shape=input_shape
    self.num_tv=num_tv
    self.rnns = nn.ModuleList([nn.RNN(input_size=self.input_shape, hidden_size=64, num_layers=1, batch_first=True) for i in range(num_tv)])
    self.fcs = nn.ModuleList([nn.Linear(64,1) for i in range(self.num_tv)])
    with torch.no_grad():
        for n in range(num_tv):
            self.rnns[n].weight_hh_l0.copy_(model_last.rnn.weight_hh_l0)
            self.rnns[n].weight_ih_l0.copy_(model_last.rnn.weight_ih_l0)
            self.rnns[n].bias_hh_l0.copy_(model_last.rnn.bias_hh_l0)
            self.rnns[n].bias_ih_l0.copy_(model_last.rnn.bias_ih_l0)
            
            self.fcs[n].weight.copy_(model_last.fc.weight)
            self.fcs[n].bias.copy_(model_last.fc.bias)
          
  def forward(self, x,h_,t_ind):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    h0=torch.tensor(h_,device=device,dtype=dtype)
    for i, l in enumerate(self.rnns):
        if i==0:
            (globals()[f"x{i}"], globals()[f"h_save{i}"]) = self.rnns[i](x[:,t_ind[i]:t_ind[i+1],:],h0)
        else:
            (globals()[f"x{i}"], globals()[f"h_save{i}"]) = self.rnns[i](x[:,t_ind[i]:t_ind[i+1],:],globals()[f"h_save{i-1}"])
    h_save=torch.clone(globals()[f"h_save{i}"])
    x=x0
    for j in range(1,i+1):
        x = torch.cat((x,globals()[f"x{j}"]),axis=1)
    x_save=torch.clone(x)
    for h, f in enumerate(self.fcs):
        globals()[f"y{h}"]=self.fcs[h](globals()[f"x{h}"])
    y=y0
    for j in range(1,i+1):
        y = torch.cat((y,globals()[f"y{j}"]),axis=1)
    y=torch.sigmoid(y)
    return y,x_save,h_save

def rnn_model(lr,n_epochs,batch_size,t_ind_ini,num_tv,X_train,y_train,X_test,y_test,model_last,verbose,device):
  dataset_train = torch.utils.data.TensorDataset(X_train,y_train)
  train_loader = torch.utils.data.DataLoader(
                                              dataset_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
  batches_train = int(X_train.shape[0])
  h_initial_train = np.zeros([1,batches_train, 64])
  batches_test = int(X_test.shape[0])
  h_initial_test = np.zeros([1,batches_test, 64]) 
  input_s=X_train.shape[2]
  model = Model(input_s,num_tv,model_last).to(device)
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  loss_dy_each=np.zeros(n_epochs)
  epoch=0
  max_out=len(X_train[0])
  new_ind=t_ind_ini.copy()
  while epoch <n_epochs:
    running_loss = 0.0
    n = 0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      model.train()
      optimizer.zero_grad()
      batches_batch = int(inputs.shape[0])
      h_initial_batch = np.zeros([1,batches_batch, 64])
      y_,_,_ = model(inputs,h_initial_batch,new_ind)

      loss=torch.tensor(np.ones(y_.shape[1]))
      for j in range(y_.shape[1]):
          loss_each = criterion(y_[:,j,0], labels)
          loss[j]=loss_each
      loss.sum().backward(retain_graph=True)
      loss_c=loss.sum()
      optimizer.step()
      running_loss += loss_c.item()
      n += 1
    loss_print=running_loss/n
    model.eval()
    y_a,x_save,h_save_train_ = model(X_train,h_initial_train,new_ind)
    loss=torch.tensor(np.ones(y_a.shape[1]))
    for j in range(y_a.shape[1]):
        loss_each = criterion(y_a[:,j,0], y_train)
        loss[j]=loss_each
    y_v,x_save_test,h_save_test_ = model(X_test,h_initial_test,new_ind)
    acc_all_=np.zeros(int(y_a.shape[1]))
    val_acc_all_=np.zeros(int(y_v.shape[1]))
    for h in range(int(y_a.shape[1])):
        acc_all_[h]=decoding._get_accuracy(y_train,y_a[:,h,0])
        val_acc_all_[h]=decoding._get_accuracy(y_test,y_v[:,h,0])
    val_accuracy_print=val_acc_all_[-1]
    accuracy_print=acc_all_[-1]

    loss_dy_each[epoch]=loss_print
    loss_val=torch.tensor(np.ones(y_v.shape[1]))
    for j in range(y_v.shape[1]):
        loss_each = criterion(y_v[:,j,0], y_test)
        loss_val[j]=loss_each
    val_loss=loss_val.sum()
    if verbose==True:
      print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss_print},val_loss = {val_loss},accuracy={accuracy_print}, val_accuracy={val_accuracy_print}")
    epoch=epoch+1

  return model,acc_all_,val_acc_all_,loss_dy_each,loss_val,y_a,y_v