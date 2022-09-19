import numpy as np
import sklearn
import torch
from torch import nn
import matplotlib.pyplot as plt
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


def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

class Model(nn.Module):
    def __init__(self,input_shape):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64,1)
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


def data_processing(data_behavior,data_control):
  if data_behavior.ndim==3: # the input is (n_trials,n_region,n_timepoint), the code works only if the input shape is like this, if not, you may have to change it.
    X=np.concatenate((data_behavior,data_control),axis=0)
    X=X.transpose((0,2,1)) 
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


def rnn_model(lr,epochs,batch_size,X_train,y_train,X_test,y_test,validation,using_all,verbose,device):
  dataset_train=torch.utils.data.TensorDataset(X_train,y_train)
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
      accuracy=get_accuracy(y_train,y_a[:,-1,0])
      if validation==True:
        y_v,x_save_val = model(X_test)
        val_accuracy=get_accuracy(y_test,y_v[:,-1,0])
      else:
        val_accuracy=None
      loss_print=running_loss/n
      if verbose==True:
        print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss_print},accuracy={accuracy}, val_accuracy={val_accuracy}")
  return model,accuracy,val_accuracy,x_save,x_save_val



# def rnn_model(lr,epochs,batch_size,X_train,y_train,X_test,y_test,device,validation,using_all,verbose):
#   input_shape=X_train.shape[2]
#   dataset_train=torch.utils.data.TensorDataset(X_train,y_train)
#   train_loader = torch.utils.data.DataLoader(
#                                               dataset_train,
#                                               batch_size=batch_size,
#                                               shuffle=True,
#                                               num_workers=0)
#   n_epochs = epochs 
#   model = Model(device,input_shape).to(device)
#   criterion = nn.BCELoss()
#   optimizer = torch.optim.Adam(model.parameters(),lr=lr)
#   for epoch in range(n_epochs):
#       running_loss = 0.0
#       n = 0
#       for i, data in enumerate(train_loader, 0):
#           inputs, labels = data
#           model.train()
#           optimizer.zero_grad()
#           y_,_ = model(inputs)
#           if using_all==True:
#             loss=torch.tensor(np.ones(y_.shape[1]))
#             for j in range(y_.shape[1]):
#                 loss_each = criterion(y_[:,j,0], labels)
#                 loss[j]=loss_each
#             loss.sum().backward(retain_graph=True)
#             loss_c=loss.sum()
#           else:
#             loss = criterion(y_[:,-1,0], y_train)
#             loss.backward(retain_graph=True)
#             loss_c=loss
#           optimizer.step()
#           running_loss += loss_c.item()
#           n += 1
#       y_a,x_save = model(X_train)
#       accuracy=get_accuracy(y_train,y_a[:,-1,0])
#       if validation==True:
#         y_v,x_save_val = model(X_test)
#         val_accuracy=get_accuracy(y_test,y_v[:,-1,0])
#       else:
#         val_accuracy=None
#       loss_print=running_loss/n
#       if verbose==True:
#         print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss_print},accuracy={accuracy}, val_accuracy={val_accuracy}")
#   return model,accuracy,val_accuracy

def svm_model(X,y,num_fold): # fit svm
  clf = svm.SVC(kernel='linear')
  scores = cross_val_score(clf, X, y, cv=num_fold)
  return scores

# def decoding_time(X,y,num_fold,num_frame,lr,epochs,batch_size,verbose):
#   # run svm & rnn at each second
#   scores_svm=[] # all scores of svm, all time*all validation
#   acc_rnn=[] # all accuracy of rnn
#   num_second= X.shape[1]
#   kf = KFold(n_splits=num_fold,random_state=None, shuffle=True)# k-fold validation
#   for i in range (int(num_second/num_frame)):# 1 second has 30 frames, for each second, we did a decoding processing.
#     X_t=X[:,i*num_frame:(i+1)*num_frame,:]
#     X_time = X_t.reshape(X_t.shape[0],X_t.shape[1]*X_t.shape[2])# reshape data only for SVM, because SVM requires 2-dim data.
#     scores = svm_model(X_time, y, num_fold)
#     acc_rnn_s=[]
#     for train_index, test_index in kf.split(X_t):
#       X_train, X_test = X_t[train_index], X_t[test_index] 
#       y_train, y_test = y[train_index], y[test_index]
#       _, _, acc=rnn_model_validation(lr,epochs,batch_size,verbose,X_train,y_train,X_test,y_test) # we report the validation accuracy of last time step to be testing accuracy.
#       acc_rnn_s.append(acc)
#     scores_svm.append(scores)
#     acc_rnn.append(np.array(acc_rnn_s))
#   return np.array(scores_svm), np.array(acc_rnn)

# def plot_decoding_time(scores_svm,acc_rnn): # may not work if different tasks
#   if len(scores_svm) != len(acc_rnn):
#     raise TypeError("svm and rnn have different length") # make sure SVM and RNN have same length of output.
#   n_time = len(scores_svm)
#   svm_avg=np.mean(scores_svm,axis=1)
#   svm_serror=np.std(scores_svm,axis=1)/(scores_svm.shape[1]**0.5) # std error
#   rnn_avg=np.mean(acc_rnn,axis=1)
#   rnn_serror=np.std(acc_rnn,axis=1)/(acc_rnn.shape[1]**0.5)

#   Y=np.linspace(0,1,12)
#   X_p=np.ones(Y.size)
#   l=15
#   fig = plt.figure(figsize=(40,20), dpi=64, facecolor='white')
#   plt.xticks(fontsize=70)
#   plt.yticks(fontsize=70)
#   axes = plt.subplot(111)
#   axes.tick_params(axis ='both', which ='both', length = 10,width=4,pad=20)

#   bwith = 3
#   ax = plt.gca()
#   ax.spines['bottom'].set_linewidth(bwith)
#   ax.spines['left'].set_linewidth(bwith)
#   ax.spines['top'].set_linewidth(bwith)
#   ax.spines['right'].set_linewidth(bwith)

#   x=np.arange(-9.5,10.5,1) 
#   #plot 
#   plt.plot( (0+0)*X_p, Y, color='black',ls='--',linewidth=6)
#   plt.text(0.1, 0.23, 'Lever Pull', fontdict={'size': 70}, rotation=90)
#   plt.text(-13.27, 0.49, 'Chance', fontdict={'size': 70}, rotation=0)
#   plt.axhline(y=0.5, color='black', linestyle='--',linewidth=6)
#   plt.errorbar(x,svm_avg,svm_serror,c='blue',marker='o', mec='blue', alpha=0.6,ms=3, mew=3,linewidth=l,elinewidth=8,label='SVM')
#   plt.errorbar(x+0.2,rnn_avg,rnn_serror,c='green',marker='o', mec='green', alpha=0.6,ms=3, mew=3,linewidth=l,elinewidth=8,label='RNN')
#   plt.title('Decoding decision choice',fontdict={'size': 110},pad=60)
#   plt.xlabel('Time from Lever Pull (s)',fontdict={'size': 110},labelpad=60)
#   plt.ylabel('Decoding Accuracy',fontdict={'size': 110},labelpad=180)
#   plt.legend(loc='best',fontsize=67,fancybox=True,shadow=True)

#   ax=plt.gca()
#   ax.xaxis.set_major_locator(MultipleLocator(2))
#   ax.xaxis.set_minor_locator(MultipleLocator(1))
#   ax.yaxis.set_major_locator(MultipleLocator(0.2))
#   ax.yaxis.set_minor_locator(MultipleLocator(0.1))

#   plt.ylim(ymin = 0.2)
#   plt.xlim(-10.5,10.5,2)

#   for i in range(0,len(scores_svm)):
#       a=svm_avg[i]
#       b=svm_serror[i]
#       n=scores_svm.shape[1]
#       t=(a-(0.5))/b
#       df=n-1
#       p = (1 - stats.t.cdf(t,df=df)) # plot the p-value
#       if (p > 0.01 and p <= 0.05):
#           plt.plot(-9.5+i,svm_avg[i]+svm_serror[i]+0.02,'*',c='black',ms=10)
#       if (p > 0.00001 and p <= 0.01):
#           plt.plot(-9.5+i,svm_avg[i]+svm_serror[i]+0.02,'*',c='black',ms=10)
#           plt.plot(-9.5+i,svm_avg[i]+svm_serror[i]+0.04,'*',c='black',ms=10)
#       if  p <= 0.00001:
#           plt.plot(-9.5+i,svm_avg[i]+svm_serror[i]+0.02,'*',c='black',ms=10)
#           plt.plot(-9.5+i,svm_avg[i]+svm_serror[i]+0.04,'*',c='black',ms=10)
#           plt.plot(-9.5+i,svm_avg[i]+svm_serror[i]+0.06,'*',c='black',ms=10)

#   for i in range(0,len(acc_rnn)):
#       a=rnn_avg[i]
#       b=rnn_serror[i]
#       n=acc_rnn.shape[1]
#       t=(a-(0.5))/b
#       df=n-1
#       p = (1 - stats.t.cdf(t,df=df))
#       if (p > 0.01 and p <= 0.05):
#           plt.plot(-9.5+i+0.2,rnn_avg[i]+rnn_serror[i]+0.02,'*',c='black',ms=10)
#       if (p > 0.00001 and p <= 0.01):
#           plt.plot(-9.5+i+0.2,rnn_avg[i]+rnn_serror[i]+0.02,'*',c='black',ms=10)
#           plt.plot(-9.5+i+0.2,rnn_avg[i]+rnn_serror[i]+0.04,'*',c='black',ms=10)
#       if  p <= 0.00001:
#           plt.plot(-9.5+i+0.2,rnn_avg[i]+rnn_serror[i]+0.02,'*',c='black',ms=10)
#           plt.plot(-9.5+i+0.2,rnn_avg[i]+rnn_serror[i]+0.04,'*',c='black',ms=10)
#           plt.plot(-9.5+i+0.2,rnn_avg[i]+rnn_serror[i]+0.06,'*',c='black',ms=10)

#   return fig
