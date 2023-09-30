# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:39:56 2018

normalization of distribution
version: 210111

@author: tadahaya
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

__all__ = ["quantile","ts_norm","consensus_sig",
           "z","z_array","madz","madz_array","robz","robz_array"]


def quantile(df,method="median"):
    """
    quantile normalization of dataframe (feature x sample)
    
    Parameters
    ----------
    df: dataframe
        a dataframe subjected to QN (feature x sample)
    
    method: str, default "median"
        determine median or mean values are employed as the template    

    """
    df_c = df.copy() # deep copy
    idx = list(df_c.index)
    col = list(df_c.columns)
    n_idx = len(idx)

    ### prepare mean/median distribution
    x_sorted = np.sort(df_c.values,axis=0)[::-1]
    if method=="median":
        temp = np.median(x_sorted,axis=1)
    else:
        temp = np.mean(x_sorted,axis=1)
    temp_sorted = np.sort(temp)[::-1]
    del x_sorted

    ### prepare reference rank list
    x_rank_T = df_c.rank(method="first").T.values.astype(int)

    ### conversion
    rank = sorted([int(v + 1) for v in range(n_idx)],reverse=True)
    converter = dict(list(zip(rank,temp_sorted)))
    converted = []
    converted_ap = converted.append
    for arr in x_rank_T:
        tra = [converter[v] for v in arr]
        converted_ap(tra)
    np_data = np.array(converted).T
    df2 = pd.DataFrame(np_data,index=idx,columns=col)
    return df2


def ts_norm(df,axis=0,ts=False):
    """
    normalization with total strength of each sample (columns)
    
    Parameters
    ----------
    df: dataframe
        a dataframe subjected to ts normalization
    
    axis: int, default 0
        determine direction of normalization, row or column
        0: normalization in column vector
        1: normalization in row vector
    
    ts: boolean, default "False"
        whether total strength array is exported or not    

    """
    if axis==0:
        norms = np.linalg.norm(df,axis=0)
        df2 = df/norms
    else:
        df = df.T
        norms = np.linalg.norm(df,axis=0)
        df2 = df/norms
        df2 = df2.T
    if ts:
        return df2,norms
    else:
        return df2


def consensus_sig(data,sep="_",position=1):
    """
    to generate consensus signature
    by linear combination with weightning Spearman correlation
    
    Parameters
    ----------
    data: a dataframe
        a dataframe to be analyzed
    
    sep: str, default "_"
        separator for sample name
        
    position: int, default 1
        indicates position of sample name such as drug    
    
    """
    col = list(data.columns)
    ind = list(data.index)
    rep = [v.split(sep)[position] for v in col]
    rep_set = list(set(rep))
    rank = data.rank()
    new_col = []
    res = []
    ap = res.append
    for r in rep_set:
        mask = [col[i] for i,v in enumerate(rep) if r==v]
        new_col += mask
        temp = data[mask].values.T # check, to_numpy()
        if len(mask) > 2:
            temp_rank = rank[mask].values.T # check, to_numpy()
            corr = np.corrcoef(temp_rank)
            corr_sum = np.sum(corr,axis=1) - 1
            corr = corr/np.c_[corr_sum]
            lst = []
            for j in range(corr.shape[0]):
                temp2 = np.delete(temp,j,axis=0)
                corr2 = np.delete(corr[j],j)
                lst.append(np.dot(corr2,temp2))
            ap(np.array(lst))
        else:
            ap(temp)
    res = np.concatenate(res,axis=0).T
    df = pd.DataFrame(res,index=ind,columns=new_col)
    df = df.loc[:,col]
    return df


def z_array(x):
    """
    to calculate z scores
    
    Parameters
    ----------
    x: a numpy array
        a numpy array to be analyzed
        
    """
    myu = np.mean(x,axis=0)
    sigma = np.std(x,axis=0,ddof=0)
    return (x - myu)/sigma    


def z(x,axis=0,drop=True):
    """
    to calculate z scores from dataframe
    the scores employ population control
    
    Parameters
    ----------
    x: a dataframe
        a dataframe to be analyzed
    
    axis: 0 or 1
        whether z scores are calculate in column or row

    drop: boolean
        whether drop inf and nan
        
    """
    if axis==0:
        myu = np.mean(x.values,axis=0)
        sigma = np.std(x.values,axis=0,ddof=0)
    else:
        myu = np.c_[np.mean(x.values,axis=1)]
        sigma = np.c_[np.std(x.values,axis=1,ddof=0)]
    with np.errstate(divide='ignore',invalid='ignore'):
        df = pd.DataFrame((x.values - myu)/sigma,index=x.index,columns=x.columns)    
    if drop:
        df = df.replace(np.inf,np.nan)
        df = df.replace(-np.inf,np.nan)
        df = df.dropna()
    return df


def madz_array(x):
    """
    to calculate MAD Z
    
    Parameters
    ----------
    x: a numpy array
        a numpy array to be analyzed
        
    """
    med = np.median(x,axis=0)
    mad = np.median(np.abs(x - med),axis=0)
    return (x - med)/(1.4826*mad)


def madz(x,axis=0,drop=True):
    """
    to calculate MAD Z from dataframe
    the scores employ population control
    
    Parameters
    ----------
    x: a dataframe
        a dataframe to be analyzed
    
    axis: 0 or 1
        whether MAD Z are calculate in column or row

    drop: boolean
        whether drop inf and nan
        
    """
    if axis==0:
        med = np.median(x.values,axis=0)
        mad = np.median(np.abs(x.values - med),axis=0)
    else:
        med = np.c_[np.median(x.values,axis=1)]
        mad = np.c_[np.median(np.abs(x.values - med),axis=1)]
    with np.errstate(divide='ignore',invalid='ignore'):
        df = pd.DataFrame((x.values - med)/(1.4826*mad),index=x.index,columns=x.columns)
    if drop:
        df = df.replace(np.inf,np.nan)
        df = df.replace(-np.inf,np.nan)
        df = df.dropna()
    return df


def robz_array(x):
    """
    to calculate robust z scores
    
    Parameters
    ----------
    x: a numpy array
        a numpy array to be analyzed
        
    """
    med = np.median(x,axis=0)
    q1,q3 = np.percentile(x,[25,75],axis=0)
    niqr = (q3-q1)*0.7413
    return (x - med)/niqr


def robz(x,axis=0,drop=True):
    """
    to calculate robust z scores from dataframe
    the scores employ population control
    
    Parameters
    ----------
    x: a dataframe
        a dataframe to be analyzed
    
    axis: 0 or 1
        whether robust z scores are calculate in rows or columns

    drop: boolean
        whether drop inf and nan
        
    """
    if axis==0:
        med = np.median(x.values,axis=0)
        q1,q3 = np.percentile(x.values,[25,75],axis=0)
    else:
        med = np.c_[np.median(x.values,axis=1)]
        q1 = np.c_[np.percentile(x.values,25,axis=1)]
        q3 = np.c_[np.percentile(x.values,75,axis=1)]
    niqr = (q3-q1)*0.7413
    with np.errstate(divide='ignore',invalid='ignore'):
        df = pd.DataFrame((x.values - med)/niqr,index=x.index,columns=x.columns)
    if drop:
        df = df.replace(np.inf,np.nan)
        df = df.replace(-np.inf,np.nan)
        df = df.dropna()
    return df