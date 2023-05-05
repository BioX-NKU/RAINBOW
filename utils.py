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
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import f1_score,cohen_kappa_score,accuracy_score,jaccard_score,roc_auc_score,average_precision_score
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
    #cudnn.deterministic = True
setup_seed(2023)
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
def roc_aupr(test_new_predict,test_new):
    return roc_auc_score(test_new_predict,test_new),average_precision_score(test_new_predict,test_new)
def f1_kappa_acc_jaccard(test_pred_label,test_label):
    return f1_score(test_pred_label,test_label,average='macro'),cohen_kappa_score(test_pred_label,test_label),accuracy_score(test_pred_label,test_label),jaccard_score(test_pred_label,test_label,average='macro')

def roc_aupr(test_new_predict,test_new):
    return roc_auc_score(test_new_predict,test_new),average_precision_score(test_new_predict,test_new)
def typicalsamling(group, typicalNDict):
    name = group.name
    n = typicalNDict[name]
    return group.sample(n=n,replace=True)
def softmax(x):
    ex = np.exp(x)
    return ex/ex.sum()