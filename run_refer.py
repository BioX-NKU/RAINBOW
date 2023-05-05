import scanpy as sc
import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
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

from utils import *
from data_processing import *
from model import *
def get_refer_components(train_data,train_label,clrefer,refer_data=None,refer_label=None):
    if clrefer:
        df_re=pd.DataFrame(train_data)
        df_re['label']=list(train_label)
        
        typicalNDict = dict(zip(list(train_label),[500]*len(train_label)))
        result = pd.DataFrame()
        for i in range(1000):
            result_ = df_re.groupby('label').apply(typicalsamling, typicalNDict).drop('label',axis=1).groupby('label').mean().reset_index()
            result = pd.concat([result,result_])
        result = shuffle(result.reset_index()).drop('index',axis=1)
        re1 = np.array(result.drop('label',axis=1))
        r0=PCA(n_components=100,svd_solver='arpack')
        r0.fit(re1)
        
        try:
            r1=PCA(n_components=100,svd_solver='arpack')
        except:
            r1=PCA(n_components=refer_data.shape[0]-1,svd_solver='arpack')
        r1.fit(refer_data)

        return r0.components_,r1.components_
    else:
        df_re=pd.DataFrame(train_data)
        df_re['label']=list(train_label)
        
        typicalNDict = dict(zip(list(train_label),[500]*len(train_label)))
        result = pd.DataFrame()
        for i in range(1000):
            result_ = df_re.groupby('label').apply(typicalsamling, typicalNDict).drop('label',axis=1).groupby('label').mean().reset_index()
            result = pd.concat([result,result_])
        result = shuffle(result.reset_index()).drop('index',axis=1)
        re1 = np.array(result.drop('label',axis=1))
        
        r0=PCA(n_components=100,svd_solver='arpack')
        r0.fit(re1)
        
        return r0.components_

names = ['ALL_blood','donor_BM0828','cisTopic','MPP_LMPP_CLP']

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
peakrate = 0.03
tfidf = tfidf3
c1=standardize


for fold in [2,3,4,5]:
    for name in names:   

        train_ = sc.read('/home/sccassuper/data/sccassuper/lisiyu/data/20230320/5fold/%s/fold%d/train.h5ad'%(name,fold))
        test_ = sc.read('/home/sccassuper/data/sccassuper/lisiyu/data/20230320/5fold/%s/fold%d/test.h5ad'%(name,fold))
        refer_ = sc.read('/home/sccassuper/data/sccassuper/lisiyu/data/20230320/5fold/%s/refer.h5ad'%name)

        X_train = train_.X
        y_train = train_.obs.cell_type
        X_test = test_.X
        y_test = test_.obs.cell_type
        X_refer = refer_.X
        y_refer = refer_.obs.cell_type
               
        
        train_data,train_label,test_data,test_label,refer_data,refer_label=data_processing(X_train,(y_train),X_test,(y_test),refer=X_refer,refer_label=y_refer,clrefer=True,peak_rate=peakrate,tfidf=tfidf)
        #获得参考集主成分
        r0,r1 = get_refer_components(train_data,train_label,True,refer_data,refer_label)


        inp=(train_data).shape[1]
        contrastlearing=my_model(epochs=100,LR=0.005,inputs=inp,temp=0.1,
                                 outputs=128,datanum=1000,batch_size=64,nw=4,clrefer=True,device=device,r0=r0,r1=r1)
        contrastlearing.fit((train_data).copy(),train_label)
        test_pred_label=contrastlearing.test_predict((test_data).copy())
        return test_pred_label
        
        