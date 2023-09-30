# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:14:45 2019

@author: tadahaya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
from collections import Counter

from numpy.random import shuffle
from scipy.special import comb
import matplotlib.pyplot as plt


def rankproduct_test(array0:np.array,array1:np.array,n0:int=None,n1:int=None,n_iter:int=10000):
    """ 
    conduct permutation test of two arrays based on rank product
    
    Parameters
    ----------
    array0, array1: numpy.array or list
        an array or list of rank
        lower means more significant (ex. 1st is the most significant)

    n0, n1: int
        indicates the No. of total features constituting the rank

    n_iter: int
        indicates the iteration number for permutation
    
    """
    k0,k1 = len(array0),len(array1)
    rounds = comb(k0 + k1,np.min((k0,k1)),exact=True)
    if rounds > n_iter:
        rounds = n_iter
    fxn = lambda x,k,n: k * np.log(n + 1) - np.sum(x)
    origin = fxn(array0,k0,n0) - fxn(array1,k1,n1)
    array = np.concatenate([array0,array1])
    res = []
    ap = res.append
    for i in range(rounds):
        shuffle(array)
        a0 = array[:k0]
        a1 = array[k0:k0 + k1]
        temp0 = fxn(a0,k0,n0)
        temp1 = fxn(a1,k1,n1)
        ap(temp0 - temp1)
    res = np.array(res)
    if origin > 0:
        temp = res[res > origin]
    elif origin < 0:
        temp = res[res < origin]
    else:
        temp = res
    return len(temp)/rounds,(origin,res,rounds)


def plot_rank(
    data:pd.DataFrame,x:str='',y:str='',order:list=[],
    dot_scaler:int=100,normalize_size:bool=True,
    hue:str='',hue_order:list=[],xlabel:str='',ylabel:str='',rotation:bool=True,limit:tuple=(),
    output:str='',figsize:tuple=(),fontsize:int=18,color:str='royalblue',frame_scaler:float=None,
    alpha:float=0.7,label:str='rank',symbol_size:int=100,
    center_method:str='mean',center_color:str='navy',center_size:float=None,center_label:str=None,
    hline:float=None,hline_color:str='navy',hline_width:float=2,hline_label:str=''
    ):
    """
    returns a bubble chart considering the count size of data
    
    Parameters
    ----------
    data: pd.DataFrame
        a dataframe containing the counts as y and the conditions as x

    normalize_count: bool
        whether the plot size is normalized by max_count
    
    """
    # prepare data
    if len(order)==0:
        order = sorted(list(set(list(data[x]))))
    tick_dict = dict(zip(order,[int(x + 1) for x in range(len(order))]))
    scaler = 1.0
    if normalize_size:
        scaler = np.max(data[y])
    processed = []
    if center_method=='mean':
        method = lambda x: np.mean(x)
    elif center_method=='median':
        method = lambda x: np.median(x)
    else:
        raise KeyError('!! Wrong key: choose mean or median !!')
    means = dict()
    for o in order:
        temp = data[data[x]==o]
        if temp.shape[0]==0:
            temp_dic = dict()
            means[o] = None
        else:
            temp_l = list(temp[y])
            temp_c = Counter(temp_l)
            val = dot_scaler * np.array(list(temp_c.values())) / scaler
            temp_dic = dict(zip(list(temp_c.keys()),val))
            means[o] = method(temp_l)
        processed.append(temp_dic)
    if center_label is None:
        center_label = center_method
    if center_size is None:
        center_size = 0.2 * dot_scaler / scaler

    # plot
    if len(figsize) > 0:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    plt.rcParams["font.size"] = fontsize
    ax = fig.add_subplot(1,1,1)
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    for o,p in zip(order,processed):
        for k,v in p.items():
            ax.scatter(tick_dict[o],k,s=v,color=color,alpha=alpha)
        ax.scatter(tick_dict[o],means[o],s=center_size,color=center_color)
    if len(limit) > 0:
        ax.set_ylim([limit[0],limit[1]])
    xlim = (0.5,tick_dict[order[-1]] + 0.5)
    ax.set_xlim([xlim[0],xlim[1]])
    ax.set_xticks([tick_dict[o] for o in order])
    ax.set_xticklabels(order)
    if hline is not None:
        ax.hlines(hline,xmin=xlim[0],xmax=xlim[1],colors=hline_color,linewidths=hline_width,label=hline_label)
    if symbol_size is None:
        symbol_size = center_size
    ax.scatter(np.nan,np.nan,s=symbol_size,color=color,alpha=alpha,label=label)
    ax.scatter(np.nan,np.nan,s=symbol_size*0.25,color=center_color,label=center_label)

    # plot config
    if frame_scaler is not None:
        plt.subplots_adjust(**frame_scaler)
    if len(xlabel) > 0:
        ax.set_xlabel(xlabel)
    if len(ylabel) > 0:
        ax.set_ylabel(ylabel)
    if rotation:
        plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # plt.tight_layout()
    if len(output) > 0:
        plt.savefig(output,bbox_inches="tight")
    plt.show()


class CorrConsistency:
    def __init__(self):
        self.similarity = pd.DataFrame()
        self.res = pd.DataFrame()


    def calc_similarity(self,ref:pd.DataFrame,obj:pd.DataFrame,spearman:bool=False,sep:str=";"):
        """
        calculate correlation between the given two data sets
        
        Parameters
        ----------
        ref,obj: dataframe
            analyzed dataframes
        
        spearman: bool
            indicates whether Spearman correlaion is employed or not

        sep: str
            indicates the separator for discriminating the columns of ref and obj
        
        """
        temp = pd.concat([obj,ref],axis=1,join="inner")
        if spearman:
            temp = temp.rank()
        temp = temp - np.mean(temp,axis=0)
        temp = temp/np.linalg.norm(temp,axis=0)
        X1 = temp.values[:,:obj.shape[1]].T
        X2 = temp.values[:,obj.shape[1]:]
        corr = np.dot(X1,X2).T
        self.similarity = pd.DataFrame(corr,index=ref.columns,columns=obj.columns)


    def calc_pval(self):
        """ calculate p values """
        col = list(self.similarity.columns)
        loc = np.mean(self.similarity.values,axis=0)
        scale = np.std(self.similarity.values,axis=0,ddof=0)
        X = (self.similarity.values - loc)/scale
        X = pd.DataFrame(X,index=self.similarity.index,columns=col)
        res = [X.at[c,c] for c in col]
        pval = [norm.sf(v) for v in res]
        self.res = pd.DataFrame({'score':res,'p value':pval},index=col)


class Consistency:
    def __init__(self):
        self.calculator = CorrConsistency()
        self.ref = pd.DataFrame()
        self.obj = pd.DataFrame()
        self.similarity = pd.DataFrame()
        self.corr = pd.DataFrame()
    

    def set_data(self,ref:pd.DataFrame=None,obj:pd.DataFrame=None):
        """
        set data
        
        Parameters
        ----------
        ref: dataframe
            a gene x sample matrix employed for the reference data set (larger one)

        obj: dataframe
            a gene x sample matrix employed for the objective data set (smaller one)
        
        """
        if (ref is None) or (obj is None):
            raise ValueError('!! ref or obj is missing: check the inputs !!')
        ref.columns = [x.lower() for x in list(ref.columns)]
        obj.columns = [x.lower() for x in list(obj.columns)]
        ol = [o for o in list(obj.columns) if o in list(ref.columns)]
        obj = obj.loc[:,ol]
        self.ref = ref
        self.obj = obj


    def calc_consistency(self,**kwargs):
        """ calculate correlation between two data sets """
        if (self.ref is None) or (self.obj is None):
            raise ValueError('!! Empty data: use set_data before calculation !!')
        self.calculator.calc_similarity(self.ref,self.obj,**kwargs)
        self.calculator.calc_pval()
        self.similarity = self.calculator.similarity
        self.corr = self.calculator.res


    def extract_rank(self):
        """
        extract rank of consistency
        default setting returns the ascending rank (the higher similarity is, the lower the ranking is)
        
        """
        col = list(self.similarity.columns)
        rank = [self.similarity.rank(ascending=False).at[c,c] for c in col]
        rank2 = [self.similarity.rank(ascending=True).at[c,c] for c in col]
        n = self.similarity.shape[0]
        frac = np.array(rank2)/n
        res = pd.DataFrame({'ascending':rank,'descending':rank2,'fractional':frac},index=col)
        return res


# def plot_rank(
#     data:pd.DataFrame,x:str='',y:str='',order:list=[],dotsize=500,normalize_count:bool=True,
#     add_data:dict=dict(),add_params:dict={'color':'grey','lw':1,'ls':'-','label':''},add_ylabel:str='',
#     hue:str='',hue_order:list=[],xlabel:str='',ylabel:str='',rotation:bool=True,limit:tuple=(),
#     legend:bool=True,output:str='',figsize:tuple=(),fontsize:int=18,color:str='royalblue',
#     alpha:float=0.7,label:str='rank',center_color:str='navy',center_size:float=50,center_label:str='mean',
#     hline:float=None,hline_color:str='navy',hline_width:float=2,hline_label:str=''
#     ):
#     """
#     returns a bubble chart considering the count size of data
    
#     Parameters
#     ----------
#     data: pd.DataFrame
#         a dataframe containing the counts as x and the conditions as y

#     normalize_count: bool
#         whether the plot size is normalized by max_count

#     add_data: dict
#         add a line plot on the right side based on the indicated dict,
#         whose keys should be correspondent to the members indicated by y
    
#     """
#     # prepare data
#     if len(order)==0:
#         order = sorted(list(set(list(data[y]))))
#     max_count = 1
#     if normalize_count:
#         max_count = np.max(data[x])
#     processed = []
#     means = dict()
#     for o in order:
#         temp = data[data[y]==o]
#         if temp.shape[0]==0:
#             temp_dic = dict()
#             means[o] = None
#         else:
#             temp_l = list(temp[x])
#             temp_c = Counter(temp_l)
#             val = list(temp_c.values())/max_count
#             temp_dic = dict(zip(list(temp_c.keys()),val))
#             means[o] = np.mean(temp_l)
#         processed.append(temp_dic)

#     # plot
#     if len(figsize) > 0:
#         fig = plt.figure(figsize=figsize)
#     else:
#         fig = plt.figure()
#     plt.rcParams["font.size"] = fontsize
#     ax = fig.add_subplot(1,1,1)
#     ax.grid(axis='y')
#     ax.set_axisbelow(True)
#     for o,p in zip(order,processed):
#         for k,v in p.items():
#             ax.scatter(o,k,s=dotsize*v,color=color,alpha=alpha)
#         ax.scatter(o,means[o],s=center_size,color=center_color)
#     if len(limit) > 0:
#         ax.set_ylim([limit[0],limit[1]])
#     if hline is not None:
#         ax.hlines(hline,xmin=order[0],xmax=order[-1],colors=hline_color,linewidths=hline_width,label=hline_label)
#     ax.scatter(np.nan,np.nan,s=dotsize,color=color,alpha=alpha,label=label)
#     ax.scatter(np.nan,np.nan,s=center_size,color=center_color,label=center_label)
#     if len(add_data) > 0:
#         ax2 = ax.twinx()
#         addval = [add_data[o] for o in order]
#         print(addval)
#         ax2.plot(order,addval,**add_params)
#         if len(add_ylabel) > 0:
#             ax2.set_ylabel(add_ylabel)

#     # plot config
#     if len(xlabel) > 0:
#         ax.set_xlabel(xlabel)
#     if len(ylabel) > 0:
#         ax.set_ylabel(ylabel)
#     if rotation:
#         plt.xticks(rotation=45)
#     if legend:
#         if len(add_data) > 0:
#             h1,l1 = ax.get_legend_handles_labels()
#             h2,l2 = ax2.get_legend_handles_labels()
#             ax.legend(h1 + h2,l1 + l2, bbox_to_anchor=(1.05, 1), loc='upper left')
#         else:
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     if len(add_data)==0:
#         ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.tight_layout()
#     if len(output) > 0:
#         plt.savefig(output,bbox_inches="tight")
#     plt.show()



# def plot_rank(
#     data:pd.DataFrame,x:str='',y:str='',order:list=[],
#     dot_scaler:int=100,normalize_count:bool=True,count:dict=dict(),
#     hue:str='',hue_order:list=[],xlabel:str='',ylabel:str='',rotation:bool=True,limit:tuple=(),
#     output:str='',figsize:tuple=(),fontsize:int=18,color:str='royalblue',frame_scaler:float=None,
#     alpha:float=0.7,label:str='rank',symbol_size:int=None,
#     center_method:str='median',center_color:str='navy',center_size:float=None,center_label:str=None,
#     hline:float=None,hline_color:str='navy',hline_width:float=2,hline_label:str=''
#     ):
#     """
#     returns a bubble chart considering the count size of data
    
#     Parameters
#     ----------
#     data: pd.DataFrame
#         a dataframe containing the counts as x and the conditions as y

#     normalize_count: bool
#         whether the plot size is normalized by max_count
    
#     """
#     # prepare data
#     if len(order)==0:
#         order = sorted(list(set(list(data[y]))))
#     scaler = dict(zip(order,500 * np.ones(len(order))/dot_scaler))
#     if normalize_count:
#         if len(count)==0:
#             raise ValueError('!! give count or turn off normalize_count !!')
#         for o in order:
#             scaler[o] = count[o]/dot_scaler
#     processed = []
#     if center_method=='mean':
#         method = lambda x: np.mean(x)
#     elif center_method=='median':
#         method = lambda x: np.median(x)
#     else:
#         raise KeyError('!! Wrong key: choose mean or median !!')
#     means = dict()
#     for o in order:
#         temp = data[data[y]==o]
#         if temp.shape[0]==0:
#             temp_dic = dict()
#             means[o] = None
#         else:
#             temp_l = list(temp[x])
#             temp_c = Counter(temp_l)
#             val = np.array(list(temp_c.values())) * scaler[o]
#             temp_dic = dict(zip(list(temp_c.keys()),val))
#             means[o] = method(temp_l)
#         processed.append(temp_dic)
#     if center_label is None:
#         center_label = center_method
#     if center_size is None:
#         center_size = dot_scaler * 0.5

#     # plot
#     if len(figsize) > 0:
#         fig = plt.figure(figsize=figsize)
#     else:
#         fig = plt.figure()
#     plt.rcParams["font.size"] = fontsize
#     ax = fig.add_subplot(1,1,1)
#     ax.grid(axis='y')
#     ax.set_axisbelow(True)
#     for o,p in zip(order,processed):
#         for k,v in p.items():
#             ax.scatter(o,k,s=v,color=color,alpha=alpha)
#         ax.scatter(o,means[o],s=center_size,color=center_color)
#     if len(limit) > 0:
#         ax.set_ylim([limit[0],limit[1]])
#     if hline is not None:
#         ax.hlines(hline,xmin=order[0],xmax=order[-1],colors=hline_color,linewidths=hline_width,label=hline_label)
#     if symbol_size is None:
#         symbol_size = center_size
#     ax.scatter(np.nan,np.nan,s=symbol_size,color=color,alpha=alpha,label=label)
#     ax.scatter(np.nan,np.nan,s=symbol_size*0.5,color=center_color,label=center_label)

#     # plot config
#     if frame_scaler is not None:
#         plt.subplots_adjust(**frame_scaler)
#     if len(xlabel) > 0:
#         ax.set_xlabel(xlabel)
#     if len(ylabel) > 0:
#         ax.set_ylabel(ylabel)
#     if rotation:
#         plt.xticks(rotation=45)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     # plt.tight_layout()
#     if len(output) > 0:
#         plt.savefig(output,bbox_inches="tight")
#     plt.show()
