import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import os
from numpy import *
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold 
import pandas as pd
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
from torch import nn
import shap


device = torch.device('cuda')
dtype = torch.float32
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

class Model(nn.Module):
  def __init__(self,input_shape):
      super(Model, self).__init__()
      self.rnn = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.fc = nn.Linear(64,1)

  def forward(self, x):
      batches = int(x.shape[0])
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      h0 = torch.zeros([1,batches, 64],device=device)
      (x, _) = self.rnn(x,h0)
      x_save=torch.clone(x)
      x = self.fc(x)
      x=torch.sigmoid(x)
  
      return x[:,:,0]

def get_shap_ori(state,X_train,X_test):
  input_shape=X_train.shape[2]
  model = Model(input_shape).cuda()
  model.load_state_dict(state)
  background1 = X_train[:100]
  background2 = X_train[-100:]
  background = torch.cat((background1,background2),axis=0)
  test_images = X_test

  e = shap.GradientExplainer(model, background)
  shap_values = e.shap_values(test_images)

  return shap_values

class Model_tv(nn.Module):
  def __init__(self,input_shape):
      super(Model_tv, self).__init__()
      self.rnn0 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn1 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn2 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn3 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn4 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn5 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn6 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn7 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn8 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      self.rnn9 = nn.RNN(input_size=input_shape, hidden_size=64, num_layers=1, batch_first=True)
      
      self.fc0 = nn.Linear(64,1)
      self.fc1 = nn.Linear(64,1)
      self.fc2 = nn.Linear(64,1)
      self.fc3 = nn.Linear(64,1)
      self.fc4 = nn.Linear(64,1)
      self.fc5 = nn.Linear(64,1)
      self.fc6 = nn.Linear(64,1)
      self.fc7 = nn.Linear(64,1)
      self.fc8 = nn.Linear(64,1)
      self.fc9 = nn.Linear(64,1)
      
          

  def forward(self, x):
      batches = int(x.shape[0])
      h_ = torch.zeros([1,batches, 64])
      h0=to_t(h_)
      (x0, h_save0) = self.rnn0(x[:,0:30,:],h0)
      (x1, h_save1) = self.rnn1(x[:,30:60,:],h_save0)
      (x2, h_save2) = self.rnn2(x[:,60:90,:],h_save1)
      (x3, h_save3) = self.rnn3(x[:,90:120,:],h_save2)
      (x4, h_save4) = self.rnn4(x[:,120:150,:],h_save3)
      (x5, h_save5) = self.rnn5(x[:,150:180,:],h_save4)
      (x6, h_save6) = self.rnn6(x[:,180:210,:],h_save5)
      (x7, h_save7) = self.rnn7(x[:,210:240,:],h_save6)
      (x8, h_save8) = self.rnn8(x[:,240:270,:],h_save7)
      (x9, h_save9) = self.rnn9(x[:,270:300,:],h_save8)
      
      h_save=torch.clone(h_save9)
      x = torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9),axis=1)
      x_save=torch.clone(x)
      y0 = self.fc0(x0)
      y1 = self.fc1(x1)
      y2 = self.fc2(x2)
      y3 = self.fc3(x3)
      y4 = self.fc4(x4)
      y5 = self.fc5(x5)
      y6 = self.fc6(x6)
      y7 = self.fc7(x7)
      y8 = self.fc8(x8)
      y9 = self.fc9(x9)
      
      y = torch.cat((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9),axis=1)
      y=torch.sigmoid(y)

      return y
def get_shap(state,X_train,X_test):
  input_shape=X_train.shape[2]
  model_tv = Model_tv(input_shape).cuda()
  model_tv.load_state_dict(state)
  background1 = X_train[:100]
  background2 = X_train[-100:]
  background = torch.cat((background1,background2),axis=0)
  test_images = X_test

  e = shap.GradientExplainer(model_tv, background)
  shap_values = e.shap_values(test_images)

  return shap_values
