# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:46:32 2019

@author: tadahaya
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from .exp2res import exp2res as e2r
from . import consistency as cons

# for OS compatibility
if os.name == "nt":
    DIR_SEP = "\\"
elif os.name == 'posix':
    DIR_SEP = "/"


class Diff():
    def __init__(self):
        self.method = MeanCentering()


    def to_meancentering(self):
        """ switch to the mean centering method """
        self.method = MeanCentering()


    def to_stochastic(self,method:str='z'):
        """
        switch to the stochastic method
        
        Parameters
        ----------
        method: str
            indicates the method for calculation of stochastic difference
        
        """
        self.method = Stochastic(method)        


    def calc(
        self,data:pd.DataFrame,batch:pd.DataFrame=None,key_compound:str='',
        vehicle_control:bool=False,control_names:list=[],drop_control:bool=True,
        intra_batch:bool=True,key_batch:str='',export_control:bool=False
        ):
        """
        calculate difference

        Parameters
        ----------
        data: dataframe
            gene x sample matrix
        
        batch: dataframe
            batch information table whose indices are correspondent with the samples of data

        key_compound: str
            indicates the key of compound names in batch information

        vehicle_control: bool
            whether vehicle control is employed in difference calculation

        control_names: list
            indicates the names of the control employed in difference calculation

        drop_control: bool
            whether control data is removed from the output or not

        intra_batch: bool
            whether difference is calculated in each batch or in whole samples

        key_batch: str
            indicates the key of batch information employed in difference calculation

        export_control: bool
            whether output includes the control columns or not
        
        """
        return self.method.calc(
            data,batch,key_compound,vehicle_control,control_names,drop_control,
            intra_batch,key_batch,export_control
            )


class MeanCentering():
    def calc(
        self,data:pd.DataFrame,batch:pd.DataFrame=None,key_compound:str='',
        vehicle_control:bool=False,control_names:list=[],drop_control:bool=True,
        intra_batch:bool=True,key_batch:str='',
        export_control:bool=False
        ):
        """ calculate difference """
        if batch is None:
            raise ValueError('!! Give batch !!')
        try:
            con_batch = batch[batch[key_compound].isin(control_names)]
        except KeyError:
            raise KeyError('!! Wrong key_compound !!')
        con_idx = set(con_batch.index)
        if vehicle_control:
            if con_batch.shape[0]==0:
                raise KeyError('!! No matched controls were found: check control_names !!')
            if intra_batch: # MC with vehicle_control and in intra-batch
                try:
                    batch_id = list(set(list(batch[key_batch])))
                except KeyError:
                    raise KeyError('!! Wrong key_batch !!')
                res = []
                for b in batch_id:
                    temp_batch = batch[batch[key_batch]==b]
                    sample = set(temp_batch.index)
                    sample_con = sample & con_idx
                    temp_data = data.loc[:,sample]
                    con_data = data.loc[:,sample_con]
                    mean = np.c_[np.mean(con_data,axis=1)]
                    res.append(temp_data - mean)
                res = pd.concat(res,axis=1,join='inner')
                res = res.loc[:,data.columns]
            else: # MC with vehicle_control and in whole
                mean = np.c_[np.mean(data.loc[:,con_idx],axis=1)]
                res = data - mean
        else:
            batch2 = batch.copy()
            data2 = data.copy()
            if drop_control:            
                batch2 = batch.drop(con_idx,axis=0)
                data2 = data.drop(con_idx,axis=1)
            if intra_batch: # MC in intra-batch
                try:
                    batch_id = list(set(list(batch2[key_batch])))
                except KeyError:
                    raise KeyError('!! Wrong key_batch !!')
                res = []
                for b in batch_id:
                    temp_batch = batch2[batch2[key_batch]==b]
                    sample = set(temp_batch.index)
                    temp_data = data2.loc[:,sample]
                    mean = np.c_[np.mean(temp_data,axis=1)]
                    res.append(temp_data - mean)
                res = pd.concat(res,axis=1,join='inner')
                res = res.loc[:,data2.columns]
                del data2,batch2
            else: # MC in whole
                temp = np.c_[np.mean(data2,axis=1)]
                res = data2 - temp
                del data2,batch2
        if export_control==False:
            try:
                res = res.drop(con_idx,axis=1)
            except KeyError:
                pass
        return res


class Stochastic():
    def __init__(self,method:str='z'):
        if method=='z':
            self.method = Z()
        elif method=='madz':
            self.method = MADZ()
        elif method=='robz':
            self.method = RobustZ()
        else:
            raise KeyError('!! Wrong method: choose z, madz, or robz !!')


    def to_z(self):
        """ switch to z """
        self.method = Z()


    def to_madz(self):
        """ switch to MAD z """
        self.method = MADZ()


    def to_robz(self):
        """ switch to robust z """
        self.method = RobustZ()   


    def calc(
        self,data:pd.DataFrame,batch:pd.DataFrame=None,key_compound:str='',
        vehicle_control:bool=False,control_names:list=[],drop_control:bool=True,
        intra_batch:bool=True,key_batch:str='',export_control:bool=False
        ):
        """ calculate difference """
        if batch is None:
            raise ValueError('!! Give batch !!')
        try:
            con_batch = batch[batch[key_compound].isin(control_names)]
        except KeyError:
            raise KeyError('!! Wrong key_compound !!')
        con_idx = set(con_batch.index)
        with np.errstate(divide='ignore',invalid='ignore'):
            if vehicle_control:
                if con_batch.shape[0]==0:
                    raise KeyError('!! No matched controls were found: check control_names !!')
                if intra_batch: # normalized values with vehicle_control and in intra-batch
                    try:
                        batch_id = list(set(list(batch[key_batch])))
                    except KeyError:
                        raise KeyError('!! Wrong key_batch !!')
                    res = []
                    for b in batch_id:
                        temp_batch = batch[batch[key_batch]==b]
                        sample = set(temp_batch.index)
                        sample_con = sample & con_idx
                        temp_data = data.loc[:,sample]
                        con_data = data.loc[:,sample_con]
                        loc = self.method.calc_loc(con_data)
                        scale = self.method.calc_scale(con_data)
                        res.append((temp_data - loc)/scale)
                    res = pd.concat(res,axis=1,join='inner')
                    res = res.loc[:,data.columns]
                else: # normalized values with vehicle_control and in whole
                    temp = data.loc[:,con_idx]
                    loc = self.method.calc_loc(temp)
                    scale = self.method.calc_scale(temp)
                    res = (data - loc)/scale
            else:
                batch2 = batch.copy()
                data2 = data.copy()
                if drop_control:       
                    batch2 = batch.drop(con_idx,axis=0)
                    data2 = data.drop(con_idx,axis=1)                
                if intra_batch: # normalized values in intra-batch
                    try:
                        batch_id = list(set(list(batch2[key_batch])))
                    except KeyError:
                        raise KeyError('!! Wrong key_batch !!')
                    res = []
                    for b in batch_id:
                        temp_batch = batch2[batch2[key_batch]==b]
                        sample = set(temp_batch.index)
                        temp_data = data2.loc[:,sample]
                        loc = self.method.calc_loc(temp_data)
                        scale = self.method.calc_scale(temp_data)
                        res.append((temp_data - loc)/scale)
                    res = pd.concat(res,axis=1,join='inner')
                    res = res.loc[:,data2.columns]
                else: # normalized values in whole
                    loc = self.method.calc_loc(data2)
                    scale = self.method.calc_scale(data2)
                    res = (data2 - loc)/scale
        if export_control==False:
            try:
                res = res.drop(con_idx,axis=1)
            except KeyError:
                pass
        return res


class Z():
    def calc_loc(self,data):
        """ calculate location parameter """
        return np.c_[np.mean(data,axis=1)]


    def calc_scale(self,data):
        """ calculate scale parameter """
        return np.c_[np.std(data,axis=1,ddof=0)]


class MADZ():
    def __init__(self):
        self.loc = None


    def calc_loc(self,data):
        """ calculate location parameter """
        self.loc = np.c_[np.median(data,axis=1)]
        return self.loc


    def calc_scale(self,data):
        """ calculate scale parameter """
        try:
            mad = np.c_[np.median(np.abs(data - self.loc),axis=1)]
            return 1.4826 * mad
        except TypeError:
            _ = self.calc_loc(data)
            mad = np.c_[np.median(np.abs(data - self.loc),axis=1)]
            return 1.4826 * mad
            

class RobustZ():
    def calc_loc(self,data):
        """ calculate location parameter """
        return np.c_[np.median(data,axis=1)]


    def calc_scale(self,data):
        """ calculate scale parameter """
        q1 = np.c_[np.percentile(data,25,axis=1)]
        q3 = np.c_[np.percentile(data,75,axis=1)]
        return 0.7413 * (q3 - q1)


def intra_dc(
    method:str='z',data:pd.DataFrame=None,batch:pd.DataFrame=None,key_compound:str='',
    vehicle_control:bool=False,control_names:list=[],drop_control:bool=True,
    intra_batch:bool=True,key_batch:str='',export_control:bool=False,spearman:bool=False,
    return_response:bool=False
    ):
    """
    calculate difference

    Parameters
    ----------
    data: dataframe
        gene x sample matrix
    
    batch: dataframe
        batch information table whose indices are correspondent with the samples of data

    key_compound: str
        indicates the key of compound names in batch information

    vehicle_control: bool
        whether vehicle control is employed in difference calculation

    control_names: list
        indicates the names of the control employed in difference calculation

    drop_control: bool
        whether control data is removed from the output or not

    intra_batch: bool
        whether difference is calculated in each batch or in whole samples

    key_batch: str
        indicates the key of batch information employed in difference calculation

    export_control: bool
        whether output includes the control columns or not

    spearman: bool
        whether intra-dataset consistency is calculated with Spearman correlation or not
    
    """
    dat = Diff()
    print(f"> employ {method} score")
    if method in {'z','madz','robz'}:
        dat.to_stochastic(method)
    compound = set(list(batch[key_compound])) - set(control_names)
    response = dat.calc(
        data,batch,key_compound,vehicle_control,control_names,
        drop_control,intra_batch,key_batch)
    if spearman:
        response = response.rank()
    corr = response.corr()
    result = []
    nega = []
    col = []
    idx = set(list(corr.index))
    for c in tqdm(compound):
        temp = batch[batch[key_compound]==c]
        temp = list(temp.index)
        tc_p = corr.loc[temp,temp]
        tc_n = corr.loc[idx - set(temp),temp]
        result.append(tc_p.values[0,1])
        nega.append(list(np.mean(tc_n,axis=1)))
        col.append(c)
    result = pd.DataFrame({'intraDC':result},index=col)
    nega = pd.DataFrame(nega,index=compound).T
    if return_response:
        return result.sort_values('intraDC',ascending=False),nega,response
    else:
        return result.sort_values('intraDC',ascending=False),nega


def ks_test(idc:pd.DataFrame,nega:pd.DataFrame,limit:tuple=None):
    """
    plot intra-dataset consistency vs its null distribution

    """
    a = idc['intraDC'].values.ravel()
    b = nega.values.ravel()
    res = stats.ks_2samp(a,b)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.rcParams['font.size'] = 14
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if limit is not None:
        ax.set_xlim(limit)
    a = pd.DataFrame({'value':a,'key':['intraDC'] * len(a)})
    b = pd.DataFrame({'value':b,'key':['null'] * len(b)})
    data = pd.concat([a,b],axis=0,join='inner')
    colordict = {'intraDC':'goldenrod','null':'grey'}
    sns.ecdfplot(data=data,x='value',hue='key',palette=colordict,lw=2)
    plt.tight_layout()
    plt.show()
    return res


def ecdf(data:np.array):
    """ prepare data for ecdf plot """
    x = np.sort(data)
    n = len(x)
    y = np.arange(1, n + 1) / n
    return x,y


def to_response(
    method:str='z',data:pd.DataFrame=None,batch:pd.DataFrame=None,key_compound:str='',
    vehicle_control:bool=False,control_names:list=[],drop_control:bool=True,
    intra_batch:bool=True,key_batch:str='',spearman:bool=False,out_dir:str='',name:str=''
    ):
    """ convert expression to response """
    # diff
    dat = Diff()
    if method in {'z','madz','robz'}:
        dat.to_stochastic(method)
    response = dat.calc(
        data,batch,key_compound,vehicle_control,control_names,
        drop_control,intra_batch,key_batch)
    if spearman:
        response = response.rank()
    
    # consensus signature
    dat2 = e2r.Profile()
    dat2.set_batch(batch)
    dat2.set_data(response)

    # 9. consensus
    dat2.consensus_sig(key_compound,show=False)
    res = dat2.summarize_sample(key_compound,key=key_compound,with_id=False,show=False)

    # export
    if len(out_dir) > 0:
        if len(name) > 0:
            res.to_pickle(out_dir + DIR_SEP + 'grp_{}.pkl'.format(name))
        else:
            res.to_pickle(out_dir + DIR_SEP + 'grp.pkl')

    return res


def inter_dc(ref:pd.DataFrame,obj:pd.DataFrame,output:str=''):
    """ calculate inter-dataset conssitency """
    dat = cons.Consistency()
    dat.set_data(ref=ref,obj=obj)
    dat.calc_consistency()
    res = dat.similarity
    if len(output) > 0:
        res.to_csv(output)
    comp = list(res.columns)
    focus = []
    null = []
    for c in comp:
        focus.append(res.at[c,c])
        temp = res[c].copy()
        temp = temp.drop(c)
        null.append(list(temp))
    focus = pd.DataFrame({'interDC':focus},index=comp)
    null = pd.DataFrame(null,index=comp).T
    return focus.sort_values('interDC',ascending=False),null


def to_response_qc(
    method:str='z',data:pd.DataFrame=None,batch:pd.DataFrame=None,key_compound:str='',
    vehicle_control:bool=False,control_names:list=[],drop_control:bool=True,
    intra_batch:bool=True,key_batch:str='',spearman:bool=False,out_dir:str='',name:str='',
    qc:list=[]
    ):
    """ convert expression to response with intraDC-based QC """
    # diff
    dat = Diff()
    if method in {'z','madz','robz'}:
        dat.to_stochastic(method)
    response = dat.calc(
        data,batch,key_compound,vehicle_control,control_names,
        drop_control,intra_batch,key_batch)
    if spearman:
        response = response.rank()
    
    # prepare null distribution
    compound = set(list(batch[key_compound])) - {'DMSO','medium','ethanol'}
    corr = response.corr()
    result = []
    nega = []
    col = []
    idx = set(list(corr.index))
    for c in tqdm(compound):
        temp = batch[batch[key_compound]==c]
        temp = list(temp.index)
        tc_p = corr.loc[temp,temp]
        tc_n = corr.loc[idx - set(temp),temp]
        result.append(tc_p.values[0,1])
        nega.append(list(np.mean(tc_n,axis=1)))
        col.append(c)
    del corr
    result = pd.DataFrame({'intraDC':result},index=col)
    nega = np.array(nega).ravel()

    # prepare the threshold from the null
    thresh = [_upper_iqr(nega,v) for v in qc]
    thresh = dict(zip(qc,thresh))

    # hold the satisfied samples
    hold = dict()
    for k,v in thresh.items():
        print(k)
        temp = result.copy()
        print(temp.shape)
        temp = temp[temp > v].dropna()
        print(temp.shape)
        hold[k] = list(temp.index)

    # consensus signature
    dat2 = e2r.Profile()
    dat2.set_batch(batch)
    dat2.set_data(response)

    # 9. consensus
    dat2.consensus_sig(key_compound,show=False)
    res = dat2.summarize_sample(key_compound,key=key_compound,with_id=False,show=False)

    # selection based on hold
    results = dict()
    for k,v in hold.items():
        results[k] = res.loc[:,v]

    # export
    if len(out_dir) > 0:
        if len(name) > 0:
            for k,v in results.items():
                temp = str(k).replace('.','')
                v.to_pickle(out_dir + DIR_SEP + 'grp_{0}_{1}.pkl'.format(name,temp))
        else:
            for k,v in results.items():
                temp = str(k).replace('.','')
                v.to_pickle(out_dir + DIR_SEP + 'grp_{0}.pkl'.format(temp))
    return results,thresh


def _upper_iqr(array:np.array,fold:float=2.0):
    """ get the outliers of array """
    q3 = np.percentile(array,75)
    q1 = np.percentile(array,25) 
    scale = q3 - q1
    return np.median(array) + fold*scale