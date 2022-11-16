import numpy as np
import decoding


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
  hh_fold=[]
  ih_fold=[]
  hh_b_fold=[]
  ih_b_fold=[]
  o_fold=[]
  o_b_fold=[]
  for w in range(num_tv):    
      hh_fold.append(model_tv.rnns[w].weight_hh_l0.cpu().detach().numpy())
      ih_fold.append(model_tv.rnns[w].weight_ih_l0.cpu().detach().numpy())
      hh_b_fold.append(model_tv.rnns[w].bias_hh_l0.cpu().detach().numpy())
      ih_b_fold.append(model_tv.rnns[w].bias_ih_l0.cpu().detach().numpy())
      o_fold.append(model_tv.fcs[w].weight.cpu().detach().numpy())
      o_b_fold.append(model_tv.fcs[w].bias.cpu().detach().numpy())

  hh_fold=np.array(hh_fold) ##### get all recurrent weights ####
  ih_fold=np.array(ih_fold) ##### get all input weights ####
  hh_b_fold=np.array(hh_b_fold) ##### get all recurrent bias ####
  ih_b_fold=np.array(ih_b_fold) ##### get all input bias ####
  o_fold=np.array(o_fold) ##### get all output weights ####
  o_b_fold=np.array(o_b_fold) ##### get all output bias ####


  ed_hh=np.ones((num_tv,num_tv))
  ed_ih=np.ones((num_tv,num_tv))
  ed_hh_b=np.ones((num_tv,num_tv))
  ed_ih_b=np.ones((num_tv,num_tv))
  ed_o=np.ones((num_tv,num_tv))
  ed_o_b=np.ones((num_tv,num_tv))
  for h in range(num_tv):
      whh=hh_fold[h]
      wih=ih_fold[h]
      bhh=hh_b_fold[h]
      bih=ih_b_fold[h]
      wo=o_fold[h]
      bo=o_b_fold[h]
      for l in range(num_tv):
          lwhh=hh_fold[l]
          lwih=ih_fold[l]
          lbhh=hh_b_fold[l]
          lbih=ih_b_fold[l]
          lwo=o_fold[l]
          lbo=o_b_fold[l]
          ed_hh[h,l]=np.linalg.norm(lwhh-whh)        ################################
          ed_ih[h,l]=np.linalg.norm(lwih-wih)        ################################
          ed_hh_b[h,l]=np.linalg.norm(lbhh-bhh)      #compute the Euclidean Distance#
          ed_ih_b[h,l]=np.linalg.norm(lbih-bih)      ################################
          ed_o[h,l]=np.linalg.norm(lwo-wo)           ################################
          ed_o_b[h,l]=np.linalg.norm(lbo-bo)         ################################
  return ed_hh,ed_ih,ed_hh_b,ed_ih_b,ed_o,ed_o_b