import numpy as np
import decoding
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.stats.multitest
from matplotlib.pyplot import MultipleLocator


def separate_output(y):
  # This function is to separate the trials according to their final prediction, not necessary to use during training# 
  y_0=[]
  y_1=[]
  for i in range(y.shape[0]):
      if y[i,-1,0]<=0.5:
          y_0.append(y[i,:,0])
      elif y[i,-1,0]>0.5:
          y_1.append(y[i,:,0])
  return np.array(y_0),np.array(y_1)

def weights_analysis(model_tv):
  num_tv=len(model_tv.rnns)
  hh=[]
  ih=[]
  hh_b=[]
  ih_b=[]
  o=[]
  o_b=[]
  for w in range(num_tv):    
      hh.append(model_tv.rnns[w].weight_hh_l0.cpu().detach().numpy())
      ih.append(model_tv.rnns[w].weight_ih_l0.cpu().detach().numpy())
      hh_b.append(model_tv.rnns[w].bias_hh_l0.cpu().detach().numpy())
      ih_b.append(model_tv.rnns[w].bias_ih_l0.cpu().detach().numpy())
      o.append(model_tv.fcs[w].weight.cpu().detach().numpy())
      o_b.append(model_tv.fcs[w].bias.cpu().detach().numpy())

  hh=np.array(hh) ##### get all recurrent weights ####
  ih=np.array(ih) ##### get all input weights ####
  hh_b=np.array(hh_b) ##### get all recurrent bias ####
  ih_b=np.array(ih_b) ##### get all input bias ####
  o=np.array(o) ##### get all output weights ####
  o_b=np.array(o_b) ##### get all output bias ####


  ed_hh=np.ones((num_tv,num_tv))
  ed_ih=np.ones((num_tv,num_tv))
  ed_hh_b=np.ones((num_tv,num_tv))
  ed_ih_b=np.ones((num_tv,num_tv))
  ed_o=np.ones((num_tv,num_tv))
  ed_o_b=np.ones((num_tv,num_tv))
  for h in range(num_tv):
      whh=hh[h]
      wih=ih[h]
      bhh=hh_b[h]
      bih=ih_b[h]
      wo=o[h]
      bo=o_b[h]
      for l in range(num_tv):
          lwhh=hh[l]
          lwih=ih[l]
          lbhh=hh_b[l]
          lbih=ih_b[l]
          lwo=o[l]
          lbo=o_b[l]
          ed_hh[h,l]=np.linalg.norm(lwhh-whh)        ################################
          ed_ih[h,l]=np.linalg.norm(lwih-wih)        ################################
          ed_hh_b[h,l]=np.linalg.norm(lbhh-bhh)      #compute the Euclidean Distance#
          ed_ih_b[h,l]=np.linalg.norm(lbih-bih)      ################################
          ed_o[h,l]=np.linalg.norm(lwo-wo)           ################################
          ed_o_b[h,l]=np.linalg.norm(lbo-bo)         ################################
  return ed_hh,ed_ih,ed_hh_b,ed_ih_b,ed_o,ed_o_b

def auac(X,time):
    area=0
    for i in range(1,len(X)):
        if X[i]<=0.5:
            area_plus=0
        elif X[i-1]<0.5 and X[i]>0.5:
            area_plus=0
        else:
            area_plus=(X[i-1]+X[i]-1)*(time/len(X))*0.5
        area=area+area_plus
    return area
def get_early(scores_all,min_t,max_t): # do multiple t test correction of the p-value at time from -9.5 to -0.5.    
    a_s= np.mean(scores_all,axis=0)
    c_s = np.std(scores_all,axis=0)/(scores_all.shape[0]**0.5)
    y_p=np.zeros(scores_all.shape[1])
    for i in range(scores_all.shape[1]):
        a=a_s[i]
        b=c_s[i]
        n=scores_all.shape[0]
        t=(a-(0.5))/b
        df=n-1
        p = (1 - stats.t.cdf(t,df=df))
        y_p[i]=p
    _,y_np,_,_=statsmodels.stats.multitest.multipletests(y_p, alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False) #multiple t-test
    early=max_t
    for i in range(len(y_np)):
        if i==int(len(y_np)-1):
            early=min_t
            break
        if y_np[len(y_np)-i-1]>0.05:
            early=max_t-(1/30)*i
            break
        
    return y_np,early

def plot_accuracy(val_ori_last,val_ori_all,val_dy_last):
    heng=np.linspace((-300)/30,(0)/30,300)
    fig=plt.figure(figsize=(10,6))
    plt.subplot(1,1,1)
    area_o=auac(np.mean(val_ori_last,axis=0),10)
    area_d=auac(np.mean(val_dy_last,axis=0),10)
    area_a=auac(np.mean(val_ori_all,axis=0),10)
    y_np_a,early_a=get_early(val_ori_all,-10,0)
    y_np_o,early_o=get_early(val_ori_last,-10,0)
    y_np_d,early_d=get_early(val_dy_last,-10,0)

    plt.plot(heng,np.mean(val_dy_last,axis=0),color='green',label='TV RNN',linewidth=5)
    plt.fill_between(heng,np.mean(val_dy_last,axis=0)+np.std(val_dy_last,axis=0)/(5**0.5),np.mean(val_dy_last,axis=0)-np.std(val_dy_last,axis=0)/(5**0.5),color='green',alpha=0.3)
    plt.plot(heng,np.mean(val_ori_last,axis=0),color='blue',label='RNN-S1',linewidth=5)
    plt.fill_between(heng,np.mean(val_ori_last,axis=0)+np.std(val_ori_last,axis=0)/(5**0.5),np.mean(val_ori_last,axis=0)-np.std(val_ori_last,axis=0)/(5**0.5),color='blue',alpha=0.3)
    plt.plot(heng,np.mean(val_ori_all,axis=0),color='red',label='RNN-S2',linewidth=5)
    plt.fill_between(heng,np.mean(val_ori_all,axis=0)+np.std(val_ori_all,axis=0)/(5**0.5),np.mean(val_ori_all,axis=0)-np.std(val_ori_all,axis=0)/(5**0.5),color='red',alpha=0.3)

    plt.scatter(early_d,1.02,marker='*',s=200,color='green')
    plt.scatter(early_o,1.02,marker='*',s=200,color='blue')
    plt.scatter(early_a,1.02,marker='*',s=200,color='red')

    plt.axhline(y=np.mean(val_dy_last,axis=0)[-1],xmin=0.98, color='green', linestyle='--',linewidth=4)
    plt.axhline(y=np.mean(val_ori_last,axis=0)[-1],xmin=0.98, color='blue', linestyle='--',linewidth=4)
    plt.axhline(y=np.mean(val_ori_all,axis=0)[-1],xmin=0.98, color='red', linestyle='--',linewidth=4)
    plt.axhline(y=0.5, color='black', linestyle='--',linewidth=2)


    plt.ylim([0.3,1.09])

    plt.tick_params(axis ='both', which ='both', length = 10,width=2,pad=10)

    bwith = 3 
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.tight_layout()

    plt.legend(loc='best',fontsize=25)
    plt.xlabel('Time from Lever Pull (s)',x=0.55,y=-0.05,fontsize=35)
    plt.ylabel('Temporal Accuracy',x=-0.05,y=0.53,fontsize=35)
    plt.suptitle('Widefield Data',x=0.53,y=1.05,fontsize=35)
    return fig

def plot_weight_ed(ed_all,Wx,Wh,Wy):
    fig=plt.figure(figsize=(10,7))
    plt.imshow(ed_all,extent=[-10,0,0,-10])
    cb=plt.colorbar(ticks=[0,1,2])
    cb.ax.tick_params(labelsize=35)
    bwith = 3
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax=plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.tight_layout()
    plt.xlabel('Time from Lever Pull (s)',x=0.55,y=-0.05,fontsize=35)
    plt.ylabel('Time from Lever Pull (s)',x=-0.05,y=0.53,fontsize=35)
    if Wx==True:
        plt.suptitle('Comparison of '+r'$W_{x}^{t}$',x=0.53,y=1.05,fontsize=35)
    elif Wh==True:
        plt.suptitle('Comparison of '+r'$W_{h}^{t}$',x=0.53,y=1.05,fontsize=35)
    else:
        plt.suptitle('Comparison of '+r'$W_{y}^{t}$',x=0.53,y=1.05,fontsize=35)
    return fig