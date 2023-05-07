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
from sklearn.decomposition import PCA
import episcanpy.api as epi
import anndata as ad                       
from sklearn.preprocessing import StandardScaler

from .utils import *
from .data_processing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
c1=standardize
class SCS(nn.Module):
    def __init__(self,inputs,outputs,device,clrefer,r0,r1=None,mid=128):
        super(SCS, self).__init__()
        self.device=device
        self.outputs=outputs
        self.clrefer=clrefer
        if self.clrefer:
            self.r0 = r0
            self.r1 = r1
            
            self.weight_tensor_clrefer=torch.Tensor(c1(np.concatenate([self.r0,self.r1]))).to(self.device)
            
        else:
            self.r0=r0
            self.weight_tensor_cl=torch.Tensor(c1(self.r0)).to(self.device)#自先验
        
        
        if self.clrefer:
            a0=self.weight_tensor_clrefer.shape[0]
        else:
            a0=self.weight_tensor_cl.shape[0]
        self.mid=mid+a0
        self.Linear1=nn.Linear(inputs, self.mid-a0).to(self.device)
        self.Linear2=nn.Linear(self.mid, outputs).to(self.device)
        self.scs=nn.CosineSimilarity(dim=-1)
        self.train_predict=False
        
        
    def forward(self, x1,x2):
        
        if self.clrefer:
            weight_tensor=self.weight_tensor_clrefer.T
        else:
            weight_tensor=self.weight_tensor_cl.T
        x1_0 = 1*F.normalize(nn.Sigmoid()(x1.to(self.device)@weight_tensor))
        x2_0 = 1*F.normalize(nn.Sigmoid()(x2.to(self.device)@weight_tensor))
        x1 = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x1)).to(self.device))
        x2 = 0.1*F.normalize(nn.Sigmoid()(self.Linear1(x2)).to(self.device))
        if self.train_predict:
            x1 = torch.concat([x1_0,x1],axis=1)
            x2 = torch.concat([x2_0,x2],axis=1)
        else:
            x1 = torch.concat([x1_0,x1],axis=2)
            x2 = torch.concat([x2_0,x2],axis=2)
        x1 = (self.Linear2(x1))
        x2 = (self.Linear2(x2))
        if self.train_predict:
            cosine=self.scs(x1.unsqueeze(1),x2.unsqueeze(0))
        else:
            cosine=self.scs(x1.unsqueeze(1),x2.unsqueeze(2))
        self.train_predict=False
        return cosine
class my_model:
    def __init__(self,inputs,device,clrefer,r0,r1=None,epochs=100,LR=0.001,number=0,outputs=128,datanum=10000,batch_size=64,temp=0.1,nw=4):
        self.device=device
        self.epochs=epochs#100
        self.LR=LR#0.001
        self.number=number#0
        self.inputs=inputs
        self.outputs=outputs
        self.clrefer=clrefer
        if self.clrefer:
            self.r0 = r0
            self.r1 = r1
            self.net = SCS(self.inputs,self.outputs,self.device,self.clrefer,self.r0,self.r1)
        else:
            self.r0 = r0
            self.net = SCS(self.inputs,self.outputs,self.device,self.clrefer,self.r0)
        self.datanum=datanum
        self.batch_size=batch_size
        self.temp=temp
        self.nw=nw
        
    def fit(self,train_data,train_label):
        biao=[]
        bq=[]
        self.train_data=train_data
        self.train_label=train_label
        self.index=list(train_label.value_counts()[train_label.value_counts()>0].index[:])
        for celltype in (self.index):
            train_data_ind=[celltype,train_data[[index for index,i in enumerate(train_label) if i ==celltype],:],[index for index,i in enumerate(train_label) if i ==celltype]]
            biao.append(train_data_ind[1])
            bq.append(train_data_ind[2])   
        self.biao=biao
        rd=[int(random.random()*1000) for i in range(self.datanum*2)]
        
        loss_fct = nn.CrossEntropyLoss().to(device)
        optimizer=torch.optim.Adam(self.net.parameters(),lr=self.LR)
        x1=torch.concat([torch.concat([torch.Tensor(i[rd[j]%i.shape[0]]) for i in self.biao]) for j in range(self.datanum)]).reshape(self.datanum,len(self.biao),self.biao[0].shape[1])
        x2=torch.concat([torch.concat([torch.Tensor(i[rd[-j-1]%i.shape[0]]) for i in self.biao]) for j in range(self.datanum)]).reshape(self.datanum,len(self.biao),self.biao[0].shape[1])
        dataIter = DataLoader([[x1[i],x2[i]] for i in range(x1.shape[0])], batch_size=self.batch_size, shuffle=True,drop_last=True,num_workers=4)
        
        for epoch in range(self.epochs):
            for idata in dataIter:
                out=self.net(idata[0].to(self.device),idata[1].to(self.device))
                #print(out.shape)
                labels = torch.arange(out.size(1)).long().to(self.device)
                loss1 = loss_fct(out/self.temp, labels.tile(self.batch_size).reshape(self.batch_size,len(self.biao)))
                loss2 = loss_fct(torch.transpose(out,1,2)/self.temp, labels.tile(self.batch_size).reshape(self.batch_size,len(self.biao)))
                loss=loss1+loss2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
    def test_predict(self,test_data,pred_newtype=False):
        self.net.train_predict=True
        cosine=self.net(torch.Tensor(test_data).to(self.device),torch.Tensor(self.train_data).to(self.device)).detach().to('cpu')
        a=pd.DataFrame(cosine.T)
        a.index=self.train_label
        a=a.groupby(['cell_type']).mean()
        my_predict=[a[i].sort_values().index[-1]for i in range(len(test_data))]#返回预测细胞类型

        if pred_newtype == False:
            return my_predict
        else:    
            a_cos = a.copy()
            a_cos_prob = a.apply(softmax,axis=0)
            entropy_cos = -a_cos.apply(lambda x: x*np.log2(x)).sum(axis=0)
            
            my_predict=[a[i].sort_values().index[-1]for i in range(len(test_data))]#返回预测细胞类型

            return my_predict,entropy_cos
    
   