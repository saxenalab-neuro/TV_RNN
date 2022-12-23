from numpy import *
from sklearn.model_selection import KFold
import decoding
import timevarying
import numpy as np
import torch

def start_train(lr,
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
                s1,
                s2,
                tv,
                save_model,
                path,
                device):
    kf = KFold(n_splits=num_fold,random_state=42, shuffle=True)
    vi_s=0
    save_val_S1=[]
    save_val_S2=[]
    save_val_tv=[]
    for train_index, test_index in kf.split(data_input):
        print('start:',vi_s)
        vi_s=vi_s+1
        X_t_train, X_t_test = data_input[train_index], data_input[test_index] 
        y_t_train, y_t_test = data_label[train_index], data_label[test_index]
        ###### run RNN-S2 ######
        if s2==True:
            model_en,acc_en_all,val_en_all,loss_en_all,en_accuracy,en_val_accuracy,en_output_train,en_output_test=decoding.rnn_model(lr,
                                                                                en_epochs,
                                                                                batch_size,
                                                                                X_t_train,
                                                                                y_t_train,
                                                                                X_t_test,
                                                                                y_t_test,
                                                                                using_all=True,
                                                                                out_all=True,
                                                                                verbose=True,
                                                                                device=device)
        else:
            en_val_accuracy=None
        if s1==True:
            ###### run RNN-S1 ######
            model_last,acc_all,val_all,loss_all,accuracy,val_accuracy,output_train,output_test=decoding.rnn_model(lr,
                                                                                epochs,
                                                                                batch_size,
                                                                                X_t_train,
                                                                                y_t_train,
                                                                                X_t_test,
                                                                                y_t_test,
                                                                                using_all=False,
                                                                                out_all=True,
                                                                                verbose=True,
                                                                                device=device)
        else:
            val_accuracy=None
        if tv==True:
            ###### run TV RNN ######
            if s1==True: # use RNN-S1 to initialize TV-RNN, if RNN-S1 existed, then run TV RNN
                model_tv,acc_all_temp,val_acc_all_temp,loss_dy_each,loss_val,tv_output_train,tv_output_test=timevarying.rnn_model(lr_tv,
                                                                                               epochs_tv,
                                                                                               batch_size_tv,
                                                                                               t_ind_ini,
                                                                                               num_tv,
                                                                                               X_t_train,
                                                                                               y_t_train,
                                                                                               X_t_test,
                                                                                               y_t_test,
                                                                                               model_last,
                                                                                               verbose=True,
                                                                                               device=device)
            else: # use RNN-S1 to initialize TV-RNN, if RNN-S1 didn't exist, then run RNN-S1 at first or custom the initialization in timevarying.py
                model_last,acc_all,val_all,loss_all,accuracy,val_accuracy,output_train,output_test=decoding.rnn_model(lr,
                                                                                epochs,
                                                                                batch_size,
                                                                                X_t_train,
                                                                                y_t_train,
                                                                                X_t_test,
                                                                                y_t_test,
                                                                                using_all=False,
                                                                                out_all=True,
                                                                                verbose=True,
                                                                                device=device)
                model_tv,acc_all_temp,val_acc_all_temp,loss_dy_each,loss_val,tv_output_train,tv_output_test=timevarying.rnn_model(lr_tv,
                                                                                               epochs_tv,
                                                                                               batch_size_tv,
                                                                                               t_ind_ini,
                                                                                               num_tv,
                                                                                               X_t_train,
                                                                                               y_t_train,
                                                                                               X_t_test,
                                                                                               y_t_test,
                                                                                               model_last,
                                                                                               verbose=True,
                                                                                               device=device)
        else:
            val_acc_all_temp=None
        if save_model==True:
            try:
                torch.save(model_en, path+'model_en_'+str(vi_s)+'.pt')
            except:
                pass
            try:
                torch.save(model_last, path+'model_last_'+str(vi_s)+'.pt')
            except:
                pass
            try:
                torch.save(model_tv, path+'model_tv_'+str(vi_s)+'.pt')
            except:
                pass
        save_val_S1.append(val_accuracy)
        save_val_S2.append(en_val_accuracy)
        save_val_tv.append(val_acc_all_temp)
    save_val_S1=np.array(save_val_S1)
    save_val_S2=np.array(save_val_S2)
    save_val_tv=np.array(save_val_tv)
    return save_val_S1,save_val_S2,save_val_tv