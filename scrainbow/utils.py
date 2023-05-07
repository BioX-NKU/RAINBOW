import scanpy as sc
import pandas as pd
import numpy as np
import os
import time
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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.utils import shuffle

def setup_seed(seed):
    """
    Set random seed.

    Parameters
    ----------
    seed
        Number to be set as random seed for reproducibility.

    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

def tfidf3(count_mat): 
    model = TfidfTransformer(smooth_idf=False, norm="l2")
    model = model.fit(np.transpose(count_mat))
    model.idf_ -= 1
    tf_idf = np.transpose(model.transform(np.transpose(count_mat)))
    return tf_idf.todense()
def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))
def standardize(x):
    return (x - np.mean(x))/(np.std(x))

def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n,replace=True)
def softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()
def get_refer_components(train_data,train_label,clrefer,refer_data=None):
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
def get_unseen_idx(test_set):
    mean = test_set.obs.groupby('leiden').mean()
    cluster_ind = [str(i) for i in list(np.where(mean.entropy_cos>0.7)[0])]
    ind = test_set.obs.leiden[test_set.obs.leiden.isin(cluster_ind)]
    percent = len(ind)/len(test_set)
    unseen_idx = [int(i) for i in list(ind.index)]
    return unseen_idx
