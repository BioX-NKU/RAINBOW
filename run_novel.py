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

def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n,replace=True)

def get_refer_components(train_data,train_label,clrefer,refer_data=None,refer_label=None):
    if clrefer:
        r0=PCA(n_components=100,svd_solver='arpack')

        r0.fit(train_data)
        try:
            r1=PCA(n_components=100,svd_solver='arpack')
        except:
            r1=PCA(n_components=refer_data.shape[0]-1,svd_solver='arpack')
        r1.fit(refer_data)
        #参考集用均值
        df_re=pd.DataFrame(refer_data)
        df_re['label']=list(refer_label)
        re1=np.array(df_re.groupby(by='label').mean())
        r2=PCA(n_components=re1.shape[0]-1,svd_solver='arpack')
        r2.fit(re1)
        return r0.components_,r1.components_,r2.components_
    else:
        df_re=pd.DataFrame(train_data)
        df_re['label']=list(train_label)
        
        typicalNDict = dict(zip(list(train_label),[500]*len(train_label)))
        result = pd.DataFrame()
        for i in range(1000):
            setup_seed(2023)
            result_ = df_re.groupby('label').apply(typicalsamling, typicalNDict).drop('label',axis=1).groupby('label').mean().reset_index()
            result = pd.concat([result,result_])
        result = shuffle(result.reset_index()).drop('index',axis=1)
        re1 = np.array(result.drop('label',axis=1))
        r0=PCA(n_components=100,svd_solver='arpack')
        r0.fit(re1)

        return r0.components_

def get_unseen_index(test_set):
    mean = test_set.obs.groupby('leiden').mean()
    cluster_ind = [str(i) for i in list(np.where(mean.entropy_cos>0.5)[0])]
    ind = test_set.obs.leiden[test_set.obs.leiden.isin(cluster_ind)]
    percent = len(ind)/len(test_data)
    unseen_idx = np.where(test_set.obs.entropy_cos>test_set.obs.entropy_cos.quantile(1-percent))[0]
    return unseen_idx
    
names = ['mca_PreFrontalCortex_62216']
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
peakrate = 0.03
tfidf = tfidf3
c1=standardize
new_lists = [['Oligodendrocytes'],
 ['Astrocytes'],
 ['Ex. neurons CPN'],
 ['Ex. neurons CThPN'],
 ['Ex. neurons SCPN'],
 ['Unknown'],
 ['SOM+ Interneurons'],
 ['Collisions'],
 ['Inhibitory neurons'],
 ['Microglia'],
 ['Endothelial I cells'],
 ['Endothelial II cells'],
 ['Podocytes'],
 ['Purkinje cells'],
 ['Macrophages']]
for name in names:
    for ii,new_list in enumerate(new_lists):
        
        for fold in [1]:
            train_ = sc.read('/home/sccassuper/data/sccassuper/lisiyu/data/20230421/new_type/leave-one-out/%s/celltype_list%d/fold%d/train.h5ad'%(name,ii+1,fold))
            test_ = sc.read('/home/sccassuper/data/sccassuper/lisiyu/data/20230421/new_type/leave-one-out/%s/celltype_list%d/fold%d/test.h5ad'%(name,ii+1,fold))
            
            X_train = train_.X
            y_train = train_.obs.cell_type
            X_test = test_.X
            y_test = test_.obs.cell_type
            

            train_data,train_label,test_data,test_label=data_processing(X_train,(y_train),X_test,(y_test),clrefer=False,peak_rate = peakrate,tfidf = tfidf)
            #获得参考集主成分
            r0= get_refer_components(train_data,train_label,clrefer=False)

            inp=(train_data).shape[1]
            contrastlearing=my_model(epochs=100,LR=0.005,inputs=inp,outputs=128,datanum=1000,batch_size=64,nw=4,clrefer=False,r0=r0,device=device)
            contrastlearing.fit((train_data).copy(),train_label)

            my_predict,prob_predict,entropy_cos,entropy_cos_prob = contrastlearing.test_predict(test_data,pred_newtype=True)

            entropy_cos = (entropy_cos-np.min(entropy_cos))/(np.max(entropy_cos)-np.min(entropy_cos))

            test_set = ad.AnnData(X=test_data,obs=pd.DataFrame({'cell_type':list(test_label),'entropy_cos':list(entropy_cos),'my_predict':list(my_predict),'prob_predict':list(prob_predict)}))
            Ncluster = 1.5*len(train_label.unique())
            epi.pp.lazy(test_set)

            epi.tl.getNClusters(test_set,Ncluster, method='leiden')
            
            
            unseen_idx = get_unseen_index(test_set)
            
            
            pred_label = np.array(my_predict).copy()

            pred_label[unseen_idx] = 'new'
            return pred_label
           