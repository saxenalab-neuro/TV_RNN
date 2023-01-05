import numpy as np
import torch
from torch import nn
import shap



class Model(nn.Module): # copy model to load
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
    
        return x



def get_shap_ori(state,X_train,X_test):
    input_shape=X_train.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(input_shape).to(device)
    model.load_state_dict(state) # load model
    background1 = X_train[:100]
    background2 = X_train[-100:]
    background = torch.cat((background1,background2),axis=0)
    test_images = X_test

    e = shap.GradientExplainer(model, background)
    shap_values = e.shap_values(test_images) # compute shap value via GradientExplainer

    return shap_values


class Model_tv(nn.Module): # copy model to load
  def __init__(self,input_shape,num_tv):
    super(Model_tv, self).__init__()
    self.input_shape=input_shape
    self.num_tv=num_tv
    self.rnns = nn.ModuleList([nn.RNN(input_size=self.input_shape, hidden_size=64, num_layers=1, batch_first=True) for i in range(num_tv)])
    self.fcs = nn.ModuleList([nn.Linear(64,1) for i in range(self.num_tv)])
          
  def forward(self, x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    batches_batch=x.shape[0]
    h_ = np.zeros([1,batches_batch, 64])
    t_ind=np.linspace(0,x.shape[1],11).astype(int)
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
    return y



def get_shap(state,X_train,X_test,num_tv):
    input_shape=X_train.shape[2]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_tv = Model_tv(input_shape,num_tv).to(device)
    model_tv.load_state_dict(state) # load model
    background1 = X_train[:100]
    background2 = X_train[-100:]
    background = torch.cat((background1,background2),axis=0)
    test_images = X_test

    e = shap.GradientExplainer(model_tv, background)
    shap_values = e.shap_values(test_images) # compute shap value via GradientExplainer
    return shap_values







