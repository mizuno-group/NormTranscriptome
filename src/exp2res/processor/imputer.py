# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:39:56 2021

imputer of data
version: 210111

@author: tadahaya
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

__all__ = ["imputer"]


def imputer(df,rep:list=[],cutoff:float=1.0,rate:float=0.8):
    """
    imputing nan and trim the values less than the indicated value
    combination of two imputation methods:
        1. replicate imputation
        2. whole imputation
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed (feature x sample)
    
    rep: list
        indicate replicates like batch
        indices should be match with the columns of df

    rate: float, default 0.8
        determine whether imupting is done or not dependent on ratio of not nan

    cutoff: float
        indicate the threshold value for trimming

    Returns
    ----------
    res: a dataframe
    
    """
    print("1. replicate imputation")
    df_res = rep_imputer(df,rep,cutoff)
    print("2. whole imputation")
    df_res = whole_imputer(df_res,rep,cutoff,rate)
    return df_res


def rep_imputer(df,rep:list=[],cutoff:float=1.0):
    """
    imputing nan and trim the values less than the indicated value
    keep features that are filled in a group although the others are NaN
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed (feature x sample)
    
    rep: list
        indicate replicates like batch
        indices should be match with the columns of df

    cutoff: float
        indicate the threshold value for trimming

    Returns
    ----------
    res: a dataframe
    
    """
    if len(rep)!=df.shape[1]:
        raise ValueError("!! Batch No. did not match data No. !!")
    df = df[df >= cutoff]
    df = df.dropna(how="all") # delete all NaN indices
    df_full = df.dropna()
    df_null = df[df.isnull().any(axis=1)]
    col = list(df_null.columns)

    rep_set = list(set(rep))
    lst = []
    for r in tqdm(rep_set):
        mask = [col[i] for i,v in enumerate(rep) if r==v]
        temp = df_null[mask].values
        if len(mask) > 1:
            nan_check = ~np.isnan(temp).any(axis=1)
            idx = np.where(nan_check)[0].tolist()
            lst += idx
    lst = list(set(lst))
    df_preserve = df_null.iloc[lst,:].fillna(cutoff)
    df_res = pd.concat([df_preserve,df_null,df_full],axis=0,join="inner").groupby(axis=0,level=0).max()
    return df_res.loc[df.index,:].dropna(how="all")


def whole_imputer(df,rep:list=[],cutoff:float=1.0,rate:float=0.9):
    """
    imputing nan and trim the values less than the indicated value
    
    Parameters
    ----------
    df: a dataframe
        a dataframe to be analyzed (feature x sample)
    
    rep: list
        indicate replicate
        should be the same length with sample

    cutoff: float
        indicate the threshold value for trimming

    rate: float, default 0.9
        determine whether imupting is done or not dependent on ratio of not nan

    Returns
    ----------
    res: a dataframe
    
    """
    df = df[df >= cutoff]
    col = list(df.columns)
    thresh = int(rate*df.shape[1])
    df = df.dropna(thresh=thresh)
    df_full = df.dropna()
    df_null = df[df.isnull().any(axis=1)]
    col = list(df_null.columns)

    batch_set = list(set(rep))
    lst = []
    ap = lst.append
    for b in tqdm(batch_set):
        mask = [col[i] for i,v in enumerate(rep) if b==v]
        temp = df[mask].T
        temp_null = df_null[mask]
        if temp.shape[1]!=1:
            temp_null = temp_null.T
            temp_null = temp_null.fillna(temp.median()).T
        ap(temp_null)
    df_null = pd.concat(lst,axis=1,join="inner").fillna(cutoff)
    df_res = pd.concat([df_full,df_null],axis=0,join="inner")
    return df_res.loc[df.index,:]
