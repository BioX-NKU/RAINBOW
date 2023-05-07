import scanpy as sc
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import episcanpy.api as epi
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import episcanpy.api as epi
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


from .utils import *
from .data_processing import *
from .model import *

def run(train_set,test_set,refer_set = None,pred_novel=False):
    
    setup_seed(2023)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    peakrate = 0.03
    tfidf = tfidf3
    c1=standardize
    
    if refer_set ==None:
        train_ = train_set.copy()
        test_ = test_set.copy()

        X_train = train_.X
        X_test = test_.X
        train_data,test_data=data_processing(X_train,X_test,clrefer=False,peak_rate = peakrate,tfidf = tfidf)
        train_label = train_.obs.cell_type
        
        #获得参考集主成分
        r0= get_refer_components(train_data,train_label,clrefer=False)

        inp=(train_data).shape[1]
        contrastlearing=my_model(epochs=100,LR=0.005,inputs=inp,outputs=128,datanum=1000,batch_size=64,nw=4,clrefer=False,r0=r0,device=device)
        contrastlearing.fit((train_data).copy(),train_label)
        if pred_novel:
            my_predict,entropy_cos = contrastlearing.test_predict(test_data,pred_newtype=True)

            entropy_cos = (entropy_cos-np.min(entropy_cos))/(np.max(entropy_cos)-np.min(entropy_cos))

            processed_test_set = ad.AnnData(X=test_data,obs=pd.DataFrame({'cell_type':list(test_label),'entropy_cos':list(entropy_cos),'my_predict':list(my_predict)}))
            Ncluster =1.5*len(train_label.unique())
            epi.pp.lazy(processed_test_set)
            epi.tl.getNClusters(processed_test_set,Ncluster, method='leiden')
            unseen_idx = get_unseen_index(processed_test_set)
            pred_label = np.array(my_predict).copy()
            pred_label[unseen_idx] = 'novel'
            
            return pred_label
        else:
            
            my_predict = contrastlearing.test_predict(test_data,pred_newtype=False)

            return my_predict
        
    else:
        train_ = train_set.copy()
        test_ = test_set.copy()
        refer_ = refer_set.copy()

        X_train = train_.X
        X_test = test_.X
        X_refer = refer_.X
        train_data,test_data,refer_data = data_processing(X_train,X_test,refer=X_refer,clrefer=True,peak_rate=peakrate,tfidf=tfidf)
        train_label = train_.obs.cell_type
        
        #获得参考集主成分
        r0,r1 = get_refer_components((train_data).copy(),train_label,True,refer_data)


        inp=(train_data).shape[1]
        contrastlearing=my_model(epochs=100,LR=0.005,inputs=inp,temp=0.1,
                                 outputs=128,datanum=1000,batch_size=64,nw=4,clrefer=True,device=device,r0=r0,r1=r1)
        contrastlearing.fit((train_data).copy(),train_label)
        
        if pred_novel:
            my_predict,entropy_cos = contrastlearing.test_predict(test_data,pred_newtype=True)

            entropy_cos = (entropy_cos-np.min(entropy_cos))/(np.max(entropy_cos)-np.min(entropy_cos))

            processed_test_set = ad.AnnData(X=test_data,obs=pd.DataFrame({'cell_type':list(test_label),'entropy_cos':list(entropy_cos),'my_predict':list(my_predict)}))
            Ncluster =1.5*len(train_label.unique())
            epi.pp.lazy(processed_test_set)
            epi.tl.getNClusters(processed_test_set,Ncluster, method='leiden')
            unseen_idx = get_unseen_index(processed_test_set)
            pred_label = np.array(my_predict).copy()
            pred_label[unseen_idx] = 'novel'
            
            return pred_label
        
        else:
            
            my_predict = contrastlearing.test_predict(test_data,pred_newtype=False)

            return my_predict