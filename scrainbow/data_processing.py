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

from .utils import *

def data_processing(train_data,test_data,peak_rate,clrefer,refer=None,tfidf=None):
    # filter peaks
    if clrefer:
        try:
            Y = train_data.T.toarray()
            Y_test = test_data.T.toarray()
            Y_refer = refer.T.toarray()
        except:
            Y=train_data.T
            Y_test=test_data.T
            Y_refer=refer.T

        filter_peak = np.sum(Y >= 1, axis=1) >= round(peak_rate*Y.shape[1])
        train_data = Y[filter_peak,:].T
        test_data = Y_test[filter_peak,:].T
        refer = Y_refer[filter_peak,:].T
        #tfidf
        if tfidf!=None:
            train_data = tfidf(train_data.T).T
            test_data = tfidf(test_data.T).T
            refer = tfidf(refer.T).T
        ss = StandardScaler()
        train_data = ss.fit_transform(train_data)
        test_data = ss.transform(test_data)
        refer = ss.fit_transform(refer)
        return train_data,test_data,refer
    else:
        try:
            Y = train_data.T.toarray()
            Y_test = test_data.T.toarray()
        except:
            Y=train_data.T
            Y_test=test_data.T

        filter_peak = np.sum(Y >= 1, axis=1) >= round(peak_rate*Y.shape[1])
        train_data = Y[filter_peak,:].T
        test_data = Y_test[filter_peak,:].T
        #tfidf
        if tfidf!=None:
            train_data = tfidf(train_data.T).T
            test_data = tfidf(test_data.T).T
          
        ss = StandardScaler()
        train_data = ss.fit_transform(train_data)
        test_data = ss.transform(test_data)
       
        return train_data,test_data

