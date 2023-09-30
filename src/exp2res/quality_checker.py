# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:14:45 2019

@author: tadahaya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
from tqdm import tqdm,trange
from scipy.special import comb
from itertools import combinations as comb


class Checker():    
    """
    a class for quality check of data
    
    """
    def __init__(self):
        self.sample = []
        self.outlier = dict()


    def sampling(self,data,n:int=5):
        """ random sampling of samples """
        self.sample = []
        col = list(data.columns).copy()
        if len(col) <= n:
            n = len(col)
        for i in range(n):
            random.shuffle(col)
            selected = col.pop()
            self.sample.append(selected)
        return self.sample


    def set_sample(self,sample:list=[]):
        """ setter of sample """
        self.sample = sample


    def update_sample(self,sample:list=[]):
        """ update samples corresponding to given samples """
        sample0 = [v for v in self.sample if v in sample]
        self.sample = sample0


    def get_sample(self):
        """ getter of sample """
        return self.sample


    def distribution(self,data,fixed:list=[],n:int=4,figsize=(6,6),fontsize=18,bins=50,
                     title:str="",alpha:float=0.2,cmap:str="tab20b"):
        """ distribution check """
        if len(fixed) > 0:
            self.sample = fixed
        elif len(self.sample)==0:
            n = np.min((n,data.shape[1]))
            self.sample = self.sampling(data,n)
        colors = plt.cm.get_cmap(cmap).colors
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.rcParams["font.size"] = fontsize
        if len(title) > 0:
            plt.title(title)
        for i,v in enumerate(self.sample):
            temp = data[v]
            ax.hist(temp,alpha=alpha,bins=bins,color=colors[i])
        plt.show()


    def check_quality_dist(self,data,rep:list=[],threshold:float=0.2,figsize=(6,6),fontsize=18,
                      title="",bins:int=0,color="navy"):
        """
        check the quality of data based on biological replicate similarity
        
        Parameters
        ----------
        data: dataframe
            dataframe to be analyzed (feature x sample)

        rep: list
            indicate replicates like batch
            indices should be match with the columns of df

        thresh: float
            indicate the threshold to drop bad quality sample

        """
        rank = data.copy().rank()
        corr = rank.corr().values
        count = Counter(rep)
        rep_uni = [k for k,v in count.items() if v > 1]
        res = []
        ap = res.append
        for r in rep_uni:
            temp = [i for i,v in enumerate(rep) if r==v]
            temp_c = corr[temp].T[temp]
            for i in range(len(temp) - 1):
                ap(temp_c[i][i + 1:])
        res = np.concatenate(res)

        # visualization
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)
        plt.rcParams["font.size"] = fontsize
        if len(title) > 0:
            plt.title(title)
        if bins==0:
            bins = int(data.shape[1]/10)
        ax.hist(res,alpha=0.5,bins=bins,color=color)
        plt.show()


    def permutation_test(self,data,rep:list=[],spearman:bool=False):
        """
        conduct quality check based on permutation
        
        Parameters
        ----------
        data: dataframe
            dataframe to be analyzed (feature x sample)

        rep: list
            indicate replicates like batch
            indices should be match with the columns of df

        spearman: bool
            whether Spearman correlation or Pearson correlation
        
        """
        temp = data.copy()
        if spearman:
            temp = temp.rank()
        corr = temp.corr().values
        n = corr.shape[0]
        for i in trange(n):
            array = corr[i].copy()
            rep_key = rep[i]
            rep_ind = [j for j,v in enumerate(rep) if v==rep_key]
            rep_val = 0
            for j in rep_ind:
                rep_val += array[j]
            for j in rep_ind:
                np.delete(array,j)            
            cb = comb(array,r=len(rep_ind))
        raise NotImplementedError


    def two_group_test(self,data,rep:list=[]):
        """
        conduct quality check based on two group test
        
        """
        raise NotImplementedError


    def corr_outlier(self,corr:pd.DataFrame,fold:float=1.5):
        """
        get the outliers of each sample correlation

        Parameters
        ----------
        corr: dataframe
            correlation matrix

        fold: float
            indicate threshold determining the outliers

        """
        col = list(corr.columns)
        res = dict()
        for c in col:
            temp = corr[c]
            temp = temp[temp < 1.0]
            upper = self._upper_std(temp.values,fold)
            res[c] = list(temp[temp > upper].index)
        self.outlier = res
        return self.outlier


    def _upper_iqr(self,array:np.array,fold:float=2.0):
        """ get the outliers of array """
        q3 = np.percentile(array,75)
        q1 = np.percentile(array,25) 
        scale = q3 - q1        
        return q3 + fold*scale


    def _upper_std(self,array:np.array,fold:float=2.0):
        """ get the outliers of array """
        loc = np.mean(array)
        scale = np.std(array,ddof=0) 
        return loc + fold*scale