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
from Decoding import decoding 
import os
from SHAP import shaping 
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable

jobid = os.getenv('SLURM_ARRAY_TASK_ID')
name_list=['IA1','IA2','IA3','IJ1','IJ2','AQ2']


name=name_list[int(jobid)]

tid=1
data_behavior=np.load('../journal_decoding/data/super/'+name+'_behavior_combine.npy',allow_pickle=True)
data_control=np.load('../journal_decoding/data/super/'+name+'_control_combine.npy',allow_pickle=True)
# data_behavior=np.load('../journal_decoding/data/super/lockout_data/'+name+'_behavior_combine_15sec.npy',allow_pickle=True)
# data_control=np.load('../journal_decoding/data/super/lockout_data/'+name+'_control_combine_15sec.npy',allow_pickle=True)
data_behavior=data_behavior.transpose(0,2,1)
data_control=data_control.transpose(0,2,1)

def get_accuracy(y_true, y_prob):
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)

lr=0.0001
epochs=5000
batch_size=5
num_fold=5
min_t=600
max_t=900
wid=10

X_tt,y_tt=decoding.data_processing(data_behavior,data_control)
X_tt=X_tt[:,min_t:max_t,:]
device = torch.device('cuda')
dtype = torch.float32
to_t = lambda array: torch.tensor(array, device=device, dtype=dtype)
from_t = lambda tensor: tensor.to("cpu").detach().numpy()

torch.random.manual_seed(1)
np.random.seed(1)

input_shape=X_tt.shape[2]
epoch_visual=[]
acc_save=[]
val_save=[]
test_save=[]
# X_t_train,X_t_test,y_t_train,y_t_test=train_test_split(X_t, y_t, test_size=0.2, random_state=1)
kf = KFold(n_splits=num_fold,random_state=42, shuffle=True) # apply k_fold to compute the average of accuracy lose
vi_s=0
x_save_all=[]
x_save_val_all=[]
x_save_test_all=[]
y_a_all=[]
y_v_all=[]
y_t_all=[]

acc_ori=[]
val_ori=[]
test_ori=[]
x_save_ori=[]
x_save_val_ori=[]
x_save_test_ori=[]
y_a_ori=[]
y_v_ori=[]
y_t_ori=[]
w_ori=[]
loss_dy_all=[]
loss_ori_all=[]
count_all=[]
for train_index, test_index in kf.split(X_tt):
    print('start:',vi_s)
    vi_s=vi_s+1
    X_tt_train, X_tt_test = X_tt[train_index], X_tt[test_index] 
    y_tt_train, y_tt_test = y_tt[train_index], y_tt[test_index]
    
    X = torch.tensor(X_tt_train).float()
    y = torch.tensor(y_tt_train).float()

    X_testt=torch.tensor(X_tt_test).float()
    y_testt=torch.tensor(y_tt_test).float()
    
    n_epochs = 2
    X_train, y_train = to_t(X), to_t(y)
    X_test, y_test= to_t(X_testt), to_t(y_testt)
    
    model_last,acc,val_acc,test_acc,x_save,x_save_val,x_save_test,y_a,y_v,y_t,count,loss_ori_each=decoding.rnn_model(lr,epochs,batch_size,X_train,y_train,X_test,y_test,
                                         validation=True,
                                         using_all=False,
                                         verbose=False,
                                         device=device)
    count_all.append(count)
    loss_ori_all.append(loss_ori_each)
    torch.save(model_last.state_dict(),'./out/model_ori_'+name+'_'+str(vi_s)+'.pth')
    # shap_value_ori=shaping.get_shap_ori(model_last.state_dict(),X_train,X_test)
    # np.save('./out/shap_ori_'+name+'_'+str(vi_s)+'.npy',shap_value_ori)
    
    y_a_ori.append(y_a.cpu().detach())
    y_v_ori.append(y_v.cpu().detach())
    y_t_ori.append(y_t.cpu().detach())
    acc_ori.append(acc)
    val_ori.append(val_acc)
    test_ori.append(test_acc)
    x_save_ori.append(x_save.cpu().detach())
    x_save_val_ori.append(x_save_val.cpu().detach())
    x_save_test_ori.append(x_save_test.cpu().detach())
    hh=model_last.rnn.weight_hh_l0
    ih=model_last.rnn.weight_ih_l0
    bi=model_last.rnn.bias_ih_l0
    bh=model_last.rnn.bias_hh_l0
    t_w=model_last.fc.weight
    b_w=model_last.fc.bias
    w_ori.append(hh.data.cpu().numpy().copy())
    w_ori.append(ih.data.cpu().numpy().copy())
    w_ori.append(bi.data.cpu().numpy().copy())
    w_ori.append(bh.data.cpu().numpy().copy())
    w_ori.append(t_w.data.cpu().numpy().copy())
    w_ori.append(b_w.data.cpu().numpy().copy())


    X_t_train,X_t_val,y_t_train,y_t_val=train_test_split(X_train, y_train, test_size=0.2, random_state=1)

    batches = int(X_t_train.shape[0])
    h_ = torch.zeros([1,batches, 64])
    batches_ = int(X_t_val.shape[0])
    hh_ = torch.zeros([1,batches_, 64])
    batches__ = int(X_test.shape[0])
    hhh_ = torch.zeros([1,batches__, 64]) 


    val_acc_all=np.ones((int(max_t-min_t)))
    test_acc_all=np.ones((int(max_t-min_t)))
    epoch_all=np.ones((int(max_t-min_t)))
    acc_all=np.ones((int(max_t-min_t)))
    x_save_fold=[]
    x_save_val_fold=[]
    x_save_test_fold=[]
    y_a_fold=[]
    y_v_fold=[]
    y_t_fold=[]
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
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
            
            with torch.no_grad():
                self.rnn0.weight_hh_l0.copy_(hh)
                self.rnn0.weight_ih_l0.copy_(ih)
                self.rnn0.bias_hh_l0.copy_(bh)
                self.rnn0.bias_ih_l0.copy_(bi)
                self.rnn1.weight_hh_l0.copy_(hh)
                self.rnn1.weight_ih_l0.copy_(ih)
                self.rnn1.bias_hh_l0.copy_(bh)
                self.rnn1.bias_ih_l0.copy_(bi)
                self.rnn2.weight_hh_l0.copy_(hh)
                self.rnn2.weight_ih_l0.copy_(ih)
                self.rnn2.bias_hh_l0.copy_(bh)
                self.rnn2.bias_ih_l0.copy_(bi)
                self.rnn3.weight_hh_l0.copy_(hh)
                self.rnn3.weight_ih_l0.copy_(ih)
                self.rnn3.bias_hh_l0.copy_(bh)
                self.rnn3.bias_ih_l0.copy_(bi)
                self.rnn4.weight_hh_l0.copy_(hh)
                self.rnn4.weight_ih_l0.copy_(ih)
                self.rnn4.bias_hh_l0.copy_(bh)
                self.rnn4.bias_ih_l0.copy_(bi)
                self.rnn5.weight_hh_l0.copy_(hh)
                self.rnn5.weight_ih_l0.copy_(ih)
                self.rnn5.bias_hh_l0.copy_(bh)
                self.rnn5.bias_ih_l0.copy_(bi)
                self.rnn6.weight_hh_l0.copy_(hh)
                self.rnn6.weight_ih_l0.copy_(ih)
                self.rnn6.bias_hh_l0.copy_(bh)
                self.rnn6.bias_ih_l0.copy_(bi)
                self.rnn7.weight_hh_l0.copy_(hh)
                self.rnn7.weight_ih_l0.copy_(ih)
                self.rnn7.bias_hh_l0.copy_(bh)
                self.rnn7.bias_ih_l0.copy_(bi)
                self.rnn8.weight_hh_l0.copy_(hh)
                self.rnn8.weight_ih_l0.copy_(ih)
                self.rnn8.bias_hh_l0.copy_(bh)
                self.rnn8.bias_ih_l0.copy_(bi)
                self.rnn9.weight_hh_l0.copy_(hh)
                self.rnn9.weight_ih_l0.copy_(ih)
                self.rnn9.bias_hh_l0.copy_(bh)
                self.rnn9.bias_ih_l0.copy_(bi)
                
                self.fc0.weight.copy_(t_w)
                self.fc0.bias.copy_(b_w)
                self.fc1.weight.copy_(t_w)
                self.fc1.bias.copy_(b_w)
                self.fc2.weight.copy_(t_w)
                self.fc2.bias.copy_(b_w)
                self.fc3.weight.copy_(t_w)
                self.fc3.bias.copy_(b_w)
                self.fc4.weight.copy_(t_w)
                self.fc4.bias.copy_(b_w)
                self.fc5.weight.copy_(t_w)
                self.fc5.bias.copy_(b_w)
                self.fc6.weight.copy_(t_w)
                self.fc6.bias.copy_(b_w)
                self.fc7.weight.copy_(t_w)
                self.fc7.bias.copy_(b_w)
                self.fc8.weight.copy_(t_w)
                self.fc8.bias.copy_(b_w)
                self.fc9.weight.copy_(t_w)
                self.fc9.bias.copy_(b_w)
                

        def forward(self, x , h_):
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

            return y,x_save,h_save      
        
    last_loss = 10000
    patience = 10
    trigger_times = 0
    model = Model().cuda()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    val_accuracy_print=0
    epoch_save=0
    loss_dy_each=np.zeros(n_epochs)
    epoch=0
    while epoch <n_epochs:
        model.train()
        optimizer.zero_grad()
        h_save_train=torch.clone(h_)
        h_save_val=torch.clone(hh_)
        h_save_test=torch.clone(hhh_)
        y_a,x_save,h_save_train_ = model(X_t_train,h_save_train)
        loss=torch.tensor(np.ones(y_a.shape[1]))
        for j in range(y_a.shape[1]):
            loss_each = criterion(y_a[:,j,0], y_t_train)
            loss[j]=loss_each
        loss.sum().backward(retain_graph=True)
        optimizer.step()
        loss_print=loss.sum()
        model.eval()
        y_v,x_save_val,h_save_val_ = model(X_t_val,h_save_val)
        acc_all=np.zeros(int(max_t-min_t))
        val_acc_all=np.zeros(int(max_t-min_t))
        for h in range(int(max_t-min_t)):
            acc_all[h]=get_accuracy(y_t_train,y_a[:,h,0])
            val_acc_all[h]=get_accuracy(y_t_val,y_v[:,h,0])
        val_accuracy_print=val_acc_all[-1]
        accuracy_print=acc_all[-1]
        loss_dy_each[epoch]=loss_print
        loss_val=torch.tensor(np.ones(y_v.shape[1]))
        for j in range(y_v.shape[1]):
            loss_each = criterion(y_v[:,j,0], y_t_val)
            loss_val[j]=loss_each
        current_loss=loss_val.sum()
        y_t,x_save_test,h_save_test_ = model(X_test,h_save_test)
        for s in range(y_t.shape[1]):
          test_acc_all[s]=get_accuracy(y_test,y_t[:,s,0])
        if current_loss > last_loss:
            trigger_times += 1

            if trigger_times >= patience:
                print('Early stopping!')
                y_t,x_save_test,h_save_test_ = model(X_test,h_save_test)
                test_accuracy=np.ones(y_t.shape[1])
                for s in range(y_t.shape[1]):
                  test_acc_all[s]=get_accuracy(y_test,y_t[:,s,0])
                epoch=n_epochs+10
        else:
          trigger_times=0
          epoch=epoch+1
        last_loss = current_loss
        # print(f"Epoch {epoch+1}/{n_epochs}, loss = {loss_print},accuracy={accuracy_print}, val_accuracy={val_accuracy_print}")
    acc_save.append(acc_all)
    val_save.append(val_acc_all)
    test_save.append(test_acc_all)
    x_save_all.append(x_save.cpu().detach())
    x_save_val_all.append(x_save_val.cpu().detach())
    x_save_test_all.append(x_save_test.cpu().detach())
    y_a_all.append(y_a.cpu().detach())
    y_v_all.append(y_v.cpu().detach())
    y_t_all.append(y_t.cpu().detach())
    loss_dy_all.append(loss_dy_each)
    torch.save(model.state_dict(),'./out/model_'+name+'_'+str(vi_s)+'.pth')
    # shap_value=shaping.get_shap(model.state_dict(),X_train,X_test)
    # np.save('./out/shap_'+name+'_'+str(vi_s)+'.npy',shap_value)
acc_save=np.array(acc_save)
val_save=np.array(val_save)
test_save=np.array(test_save)
loss_dy_all=np.array(loss_dy_all)
loss_ori_all=np.array(loss_ori_all)

np.save('./out/count_all_'+name+'_'+str(tid)+'.npy',np.array(count_all))
np.save('./out/w_ori_'+name+'_'+str(tid)+'.npy',np.array(w_ori))
np.save('./out/x_save_all_'+name+'_'+str(tid)+'.npy',np.array(x_save_all))
np.save('./out/x_save_val_all_'+name+'_'+str(tid)+'.npy',np.array(x_save_val_all))
np.save('./out/x_save_test_all_'+name+'_'+str(tid)+'.npy',np.array(x_save_test_all))
np.save('./out/y_a_all_'+name+'_'+str(tid)+'.npy',np.array(y_a_all))
np.save('./out/y_v_all_'+name+'_'+str(tid)+'.npy',np.array(y_v_all))
np.save('./out/y_t_all_'+name+'_'+str(tid)+'.npy',np.array(y_t_all))
np.save('./out/x_save_ori_'+name+'_'+str(tid)+'.npy',np.array(x_save_ori))
np.save('./out/x_save_val_ori_'+name+'_'+str(tid)+'.npy',np.array(x_save_val_ori))
np.save('./out/x_save_test_ori_'+name+'_'+str(tid)+'.npy',np.array(x_save_test_ori))
np.save('./out/y_a_ori_'+name+'_'+str(tid)+'.npy',np.array(y_a_ori))
np.save('./out/y_v_ori_'+name+'_'+str(tid)+'.npy',np.array(y_v_ori))
np.save('./out/y_t_ori_'+name+'_'+str(tid)+'.npy',np.array(y_t_ori))
np.save('./save_plot/loss_dy_all_'+name+'_'+str(tid)+'.npy',loss_dy_all)
plt.plot(np.mean(loss_dy_all,axis=0))
plt.savefig('./save_plot/loss_mean_'+name+'_'+str(tid)+'.png')
plt.close()

np.save('./save_plot/loss_ori_all_'+name+'_'+str(tid)+'.npy',loss_ori_all)
plt.plot(np.mean(loss_ori_all,axis=0))
plt.savefig('./save_plot/loss_ori_mean_'+name+'_'+str(tid)+'.png')
plt.close()

np.save('./save_plot/test_'+name+'_'+str(tid)+'.npy',test_save)
plt.plot(np.mean(test_save,axis=0))
plt.savefig('./save_plot/test_mean_'+name+'_'+str(tid)+'.png')
plt.close()

np.save('./save_plot/val_'+name+'_'+str(tid)+'.npy',val_save)
plt.plot(np.mean(val_save,axis=0))
plt.savefig('./save_plot/val_mean_'+name+'_'+str(tid)+'.png')
plt.close()

np.save('./save_plot/acc_'+name+'_'+str(tid)+'.npy',acc_save)
plt.plot(np.mean(acc_save,axis=0))
plt.savefig('./save_plot/acc_mean_'+name+'_'+str(tid)+'.png')
plt.close()


np.save('./save_plot/test_ori_'+name+'_'+str(tid)+'.npy',np.array(test_ori))
plt.plot(np.mean(np.array(test_ori),axis=0))
plt.savefig('./save_plot/test_mean_ori_'+name+'_'+str(tid)+'.png')
plt.close()

np.save('./save_plot/val_ori_'+name+'_'+str(tid)+'.npy',np.array(val_ori))
plt.plot(np.mean(np.array(val_ori),axis=0))
plt.savefig('./save_plot/val_mean_ori_'+name+'_'+str(tid)+'.png')
plt.close()

np.save('./save_plot/acc_ori_'+name+'_'+str(tid)+'.npy',np.array(acc_ori))
plt.plot(np.mean(np.array(acc_ori),axis=0))
plt.savefig('./save_plot/acc_mean_ori_'+name+'_'+str(tid)+'.png')
plt.close()
