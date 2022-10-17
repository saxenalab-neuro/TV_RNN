import numpy as np
import sklearn
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold 
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import svm
import pandas as pd
import seaborn as sns
import random
import os
from scipy import stats
from matplotlib.pyplot import MultipleLocator
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Model(nn.Module):
    def __init__(self,input_shape):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64,1)
        # torch.nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        # torch.nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        # torch.nn.init.xavier_uniform(self.rnn.bias_hh_l0)
        # torch.nn.init.xavier_uniform(self.rnn.bias_ih_l0)
        self.rnn.weight_hh_l0.data.fill_(0)
        self.rnn.weight_ih_l0.data.fill_(0)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.bias_ih_l0.data.fill_(0)

    def forward(self, x):
        batches = int(x.shape[0])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h0 = torch.zeros([1,batches, 64],device=device)
        (x, _) = self.rnn(x,h0)
        x_save=torch.clone(x)
        x = self.fc(x)
        x=torch.sigmoid(x)
    
        return x,x_save

def _get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


def data_processing(data_behavior,data_control):
  if data_behavior.ndim==3: # the input is (n_trials,n_region,n_timepoint), the code works only if the input shape is like this, if not, you may have to change it.
    X=np.concatenate((data_behavior,data_control),axis=0)
    y=np.concatenate((np.ones(data_behavior.shape[0]),np.zeros(data_control.shape[0]))) # assign class

    #normalize
    X_R= X.reshape(-1,X.shape[1]*X.shape[2])
    normal_X = preprocessing.normalize(X_R)
    n_X=normal_X.reshape(X.shape[0],X.shape[1],X.shape[2])
    X=n_X

    #nonnan
    X_nonnan=X[~np.isnan(X)]
    X=X_nonnan.reshape((X.shape[0],X.shape[1],-1))
  else:
    raise TypeError("The input should have 3 dimensions")

  return X,y

def data_processing_no_normal(data_behavior,data_control):
  if data_behavior.ndim==3: # the input is (n_trials,n_region,n_timepoint), the code works only if the input shape is like this, if not, you may have to change it.
    X=np.concatenate((data_behavior,data_control),axis=0)
    X=X.transpose((0,2,1)) 
    y=np.concatenate((np.ones(data_behavior.shape[0]),np.zeros(data_control.shape[0]))) # assign class
  else:
    raise TypeError("The input should have 3 dimensions")

  return X,y

def do_pca(n_components,data_behavior,data_control):
    X=np.concatenate((data_behavior,data_control),axis=0)
    pca=PCA(n_components=n_components)
    X=X.transpose((0,2,1))
    X_reshape=X.reshape(X.shape[0]*X.shape[1],-1)
    X_reshape_pca=pca.fit_transform(X_reshape)
    ex=pca.explained_variance_ratio_
    X_pca=X_reshape_pca.reshape(X.shape[0],X.shape[1],-1)
    X_t=X_pca.transpose((0,2,1))
    data_behavior_pca=X_t[:data_behavior.shape[0]]
    data_control_pca=X_t[data_behavior.shape[0]:]
    return data_behavior_pca,data_control_pca,ex

def do_nmf(n_components,data_behavior,data_control):
    X=np.concatenate((data_behavior,data_control),axis=0)
    if np.min(X)<0:
      X=X-np.min(X)
    nmf=NMF(n_components=n_components)
    X=X.transpose((0,2,1))
    X_reshape=X.reshape(X.shape[0]*X.shape[1],-1)
    X_reshape_nmf=nmf.fit_transform(X_reshape)
    ex=nmf.reconstruction_err_
    X_nmf=X_reshape_nmf.reshape(X.shape[0],X.shape[1],-1)
    X_t=X_nmf.transpose((0,2,1))
    data_behavior_pca=X_t[:data_behavior.shape[0]]
    data_control_pca=X_t[data_behavior.shape[0]:]
    return data_behavior_pca,data_control_pca,ex

def rnn_model(lr,epochs,batch_size,X_train,y_train,X_test,y_test,using_all,out_all,verbose,device):
  dataset_train = torch.utils.data.TensorDataset(X_train,y_train)
  train_loader = torch.utils.data.DataLoader(
                                              dataset_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=0)
  n_epochs = epochs 
  input_shape=X_train.shape[2]
  model = Model(input_shape).to(device)
  criterion = nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  acc_all=np.ones(n_epochs)
  val_all=np.ones(n_epochs)
  loss_all=np.ones(n_epochs)
  for epoch in range(n_epochs):
      running_loss = 0.0
      n = 0
      for i, data in enumerate(train_loader, 0):
          inputs, labels = data
          model.train()
          optimizer.zero_grad()
          y_,_ = model(inputs)
          if using_all==True:
            loss=torch.tensor(np.ones(y_.shape[1]))
            for j in range(y_.shape[1]):
                loss_each = criterion(y_[:,j,0], labels)
                loss[j]=loss_each
            loss.sum().backward(retain_graph=True)
            loss_c=loss.sum()
          else:
            loss = criterion(y_[:,-1,0], labels)
            loss.backward(retain_graph=True)
            loss_c=loss
          optimizer.step()
          running_loss += loss_c.item()
          n += 1
      y_a,x_save = model(X_train)
      accuracy=_get_accuracy(y_train,y_a[:,-1,0])
      y_v,x_save_val = model(X_test)
      val_accuracy=_get_accuracy(y_test,y_v[:,-1,0])
      loss_print=running_loss/n
      if verbose==True:
        print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss_print},accuracy={accuracy}, val_accuracy={val_accuracy}")
      acc_all[epoch]=accuracy
      val_all[epoch]=val_accuracy
      loss_all[epoch]=loss_print

  y_a,x_save = model(X_train)
  # print(y_a.shape)
  if out_all==True:
    accuracy=np.ones(y_a.shape[1])
    for h in range(y_a.shape[1]):
      accuracy[h]=_get_accuracy(y_train,y_a[:,h,0])
  else:
    accuracy=_get_accuracy(y_train,y_a[:,-1,0])
  y_v,x_save_val = model(X_test)
  if out_all==True:
    val_accuracy=np.ones(y_v.shape[1])
    for s in range(y_v.shape[1]):
      val_accuracy[s]=_get_accuracy(y_test,y_v[:,s,0])
  else:
    val_accuracy=_get_accuracy(y_test,y_v[:,-1,0])
  return model,acc_all,val_all,loss_all,accuracy,val_accuracy,y_a,y_v

def run_svm(X_t_train,X_t_test,y_t_train,y_t_test,n_C,kernel):
  clf=svm.SVC(kernel=kernel,C=n_C)
  X_t_train_reshape=X_t_train.reshape(X_t_train.shape[0],-1)
  X_t_test_reshape=X_t_test.reshape(X_t_test.shape[0],-1)
  print(X_t_train_reshape.shape)
  clf.fit(X_t_train_reshape,y_t_train)
  y_predict=clf.predict(X_t_train_reshape)
  acc=accuracy_score(y_predict,y_t_train)
  y_pre=clf.predict(X_t_test_reshape)
  val=accuracy_score(y_pre,y_t_test)
    
  # lda = LinearDiscriminantAnalysis()
  # lda.fit(X_t_train_reshape,y_t_train)
  # acc_lda=lda.score(X_t_train_reshape,y_t_train)
  # val_lda=lda.score(X_t_test_reshape,y_t_test)

  return acc,val
  # ,acc_lda,val_lda
