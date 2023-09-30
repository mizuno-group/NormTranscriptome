# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:14:45 2019

@author: tadahaya
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

from . import processor as pr

class Batch():    
    """
    a class for batch data handling
    
    """
    def __init__(self):
        self.data = pd.DataFrame()
        self.ID = []
        self.exp_batch = []
        self.expected_analog = ""


    # setter
    def set_data(self,df):
        """
        load batch data

        Parameters
        ----------
        df: dataframe
            set batch data (sample x batch)
            index should be primary ID for data

        """
        self.data = df
        self.ID = list(self.data.index)


    def set_id(self,id_list:list=[]):
        """ set ID for batch data """
        self.ID = id_list
        self.data.index = self.ID


    def set_exp_batch(self,exp_batch:list=[],sorting:bool=True):
        """
        set the keys indicating the experimental batches

        Parameters
        ----------
        exp_batch: list
            a list of keys indicating the experimental batches
            should be 

        sorting: bool
            whether batches are sorted by the number of contents
            the order affects the order of batch correction with Combat

        """
        if self.data.shape[0]==0:
            raise ValueError("!! Set batch before this process !!")
        self.exp_batch = exp_batch
        if len(self.exp_batch) > 1:
            if sorting:
                n = [len(set(list(self.data[v]))) for v in self.exp_batch]
                united = sorted(list(zip(n,self.exp_batch)))
                united = zip(*united)
                self.exp_batch = list(united)[1]          


    def set_expected_analog(self,key:str):
        """ set the key indicating biologically expected analogues """
        self.expected_analog = key


    def get_whole_batch(self):
        """ getter of whole batch data """
        return self.data


    def get_id(self):
        """ getter of ID """
        return list(self.data.index)


    # function
    def check_batch(self,head:int=10):
        """
        check batch information
        
        """
        self.data.head(head)

    
    def add_batch(self,key:str="new_batch",value:list=[]):
        """
        add and update batch data

        Parameters
        ----------
        key: str
            new batch name

        value: list
            new batch data
        
        """
        if len(value)!=self.data.shape[0]:
            raise ValueError("!! Length of given batch data did not match data size !!")
        self.data[key] = value


    def extract_batch(self,key:str="",data=None):
        """
        extract batch data indicated by the given key

        Parameters
        ----------
        key: str
            indicate the key for extraction

        data: dataframe
            indicate the corresponding dataframe (feature x sample) for exact matching

        """
        if len(key)==0:
            raise ValueError("!! Indicate key for selecting batch to be extracted !!")
        if data is None:
            print("!! CAUTION: data is not indicated: Careful to the correspondence between data and batch !!")
            return list(self.data[key])
        else:
            if type(data)!=type(self.data):
                raise TypeError("!! data should be given as a df !!")
            temp_batch = self.adjust2data(data)
            sample = list(data.columns)
            df = temp_batch.loc[sample,:]
            return list(df[key])


    def extract_data(self,data=None,key:str="",match:list=[],include:list=[]):
        """
        extract data based on the indicated batch information

        Parameters
        ----------
        data: dataframe
            indicate the corresponding dataframe (feature x sample) for exact matching

        key: str
            indicate the key for extraction

        match, include: list
            indicate keywords for selecting data
            match: whetehr perfectly match
            include: whether include the keyword

        """
        if len(key)==0:
            raise ValueError("!! Indicate key for selecting batch to be extracted !!")
        if data is None:
            raise ValueError("!! Indicate dataframe to be processed !!")
        if type(data)!=type(self.data):
            raise TypeError("!! data should be given as a df !!")
        sample = list(data.columns)
        batch_df = self.data.loc[sample,:]
        batch = list(batch_df[key])
        selected = []
        if len(match) > 0:
            for m in match:
                temp = [i for i,b in enumerate(batch) if b==m]
                selected += temp
        if len(include) > 0:
            for inc in include:
                temp = [i for i,b in enumerate(batch) if inc in b]
                selected += temp
        selected = list(set(selected))
        selected = list(batch_df.iloc[selected,:].index)
        selected = [v for v in sample if v in selected]
        return data[selected]


    def adjust2data(self,data):
        """
        adjust batch information to the given data

        Parameters
        ----------
        data: dataframe
            feature x sample dataframe
        
        """
        sample0 = list(data.columns)
        sample = list(self.data.index)
        sample = [v for v in sample0 if v in sample]
        return self.data.loc[sample,:]


    def combine_batch(self,order:list=[],new_name:str="combined_batch",sep:str="_"):
        """
        combine batch with the indicated separator and create a new batch

        Parameters
        ----------
        order: list
            a list indicating the combined batches

        new_name: str
            indicate the newly generated batch name

        sep: str
            indicate the separator for bridging each batch

        """
        temp = []
        for v in order:
            try:
                temp.append(list(self.data[v]))
            except KeyError:
                raise KeyError("!! {} is not in batch names: check 'order' !!")
        temp = [list(map(lambda x: x[i],temp)) for i in range(self.data.shape[0])]
        res = []
        for t in temp:
            new = str(t[0])
            for e in t[1:]:
                new += "_" + str(e)
            res.append(new)
        self.add_batch(new_name,res)


    def rename_data(self,data,key:str="",replace:bool=False,to_tail:bool=True,sep:str="_"):
        """
        rename data based on batch data

        Parameters
        ----------
        data: dataframe
            a dataframe to be renamed (feature x sample)
        
        key: str
            indicate the key for adding batch

        replace: bool
            whether original name is replaced or not

        to_tail: bool
            whether batch names are added to the tail or the head of the original names

        sep: str
            indicate the separator for bridging the original names and batch values

        """
        dic = dict(zip(self.ID,list(self.data[key])))
        data_col = list(data.columns)
        try:
            conv = [dic[v] for v in data_col]
        except KeyError:
            raise KeyError("!! data columns and batch index should be correspondent !!")
        temp = data.copy()
        if replace:
            temp.columns = conv
        else:
            if to_tail:
                new = [str(v) + sep + str(w) for v,w in zip(data_col,conv)]
            else:
                new = [str(v) + sep + str(w) for v,w in zip(conv,data_col)]
            temp.columns = new
        return temp


    def delete_batch(self,key:str=""):
        """ delete the indicated batch """
        self.data = self.data.loc[:,~self.data.columns.str.match(key)]


    def add_expected_analog(self,key:list,prefix:str="rep",sep:str=";"):
        """
        add a column indicating biologically expected analogues to batch information
        
        Parameters
        ----------
        key: list
            a list composed of the keys for constructing biologically expected analogues
            such as concentration and compound names

        prefix: str
            indicates the prefix of the new column

        sep: str
            indicates a separator for creating the new column
        
        """
        data = self.data.copy()
        temp = data[key].values
        temp = [str(v) for v in temp]
        ele = list(set(temp))
        dic = dict(zip(ele,list(range(len(ele)))))
        res = [dic[t] for t in temp]
        rep_title = prefix
        for k in key:
            rep_title += "{0}{1}".format(sep,k)
        data[rep_title] = res
        self.data = data
        self.expected_analog = rep_title


    def correct_batch(
            self,data:pd.DataFrame,exp_batch:list=[],pre_quantile:bool=True,
            method:str='median',parametric:bool=True,sorting:bool=True,ref_key:str=None
            ):
        """
        batch correction with pycombat
        refer to https://epigenelabs.github.io/pyComBat/

        Paremeters
        ----------
        data: dataframe
            a dataframe to be corrected
            feature x sample matrix, whose samples correspond to IDs of batch
    
        exp_batch: list
            indicate the name of column of experimental batch in batch data

        pre_quantile: bool
            whether quantile normalization is done before Combat
            this process will be done for each batch

        method: str, default "median"
            determine median or mean values are employed as the template    

        parametric: bool
            indicate whether parametric or non-parametric estimation is employed

        sorting: bool
            whether batches are sorted by the number of contents
            the order affects the order of batch correction with Combat

        ref_key: str
            indicatest the batch ID
            whose mean and scale are employed for general mean and scale in integration
            this argument is mainly utilized in CombinedProfile class

        """
        if parametric==False:
            print("*** nonparametric estimation may take long time ***")
        if len(exp_batch) > 0:
            self.set_exp_batch(exp_batch,sorting)
        else:
            if len(self.exp_batch)==0:
                raise ValueError("!! Indicate the keys for experimental batches !!")
        temp = data.copy()
        temp_batch = self.adjust2data(temp)
        for batch in tqdm(self.exp_batch):
            batches = set(list(temp_batch[batch]))
            conved = []
            small = [] # data with n < 3 is not subjected to Combat due to cov calculation
            for b in batches:
                temp_idx = list(temp_batch[temp_batch[batch]==b].index)
                temp_data = temp.loc[:,temp_idx]
                if pre_quantile:
                    temp_data = pr.quantile(temp_data,method)
                conved.append(temp_data)
                if len(temp_idx) < 3:
                    small += temp_idx
            conved = pd.concat(conved,join='inner',axis=1)
            conved_trim = conved.drop(columns=small)
            conved_trim = self._correct_batch(conved_trim,batch,parametric,ref_key)
            if len(small)==1:
                temp = conved_trim
                temp[small[0]] = conved[small]
            else:
                temp = pd.concat([conved_trim,conved[small]],axis=1,join='inner')
        return temp.loc[:,data.columns]


    def batch_quantile(self,data:pd.DataFrame,batch:str,method:str='median'):
        """
        conduct quantile normalization separately in the indicated batch
        
        Parameters
        ----------
        data: pd.DataFrame
            dataframe to be normalized
            the columns should be correspondent to the IDs of batch

        batch: str
            indicates the batch name

        method: str
            indicates the method for quantile normalization

        """
        temp_batch = self.adjust2data(data)
        try:
            batches = set(list(temp_batch[batch]))
        except KeyError:
            raise KeyError('!! Wrong batch indication: check batch augument !!')
        conved = []
        for b in batches:
            temp_idx = list(temp_batch[temp_batch[batch]==b].index)
            temp = data.loc[:,temp_idx]
            temp = pr.quantile(temp,method)
            conved.append(temp)
        conved = pd.concat(conved,join='inner',axis=1)
        return conved.loc[:,data.columns]
        

    def _correct_batch(self,data:pd.DataFrame,batch:str,parametric:bool,ref_key:str=None):
        """ run Combat """
        temp_batch = self.adjust2data(data)
        batch_val = list(temp_batch[batch])
        uni = sorted(list(set(batch_val)))
        if ref_key is not None:
            try:
                uni.remove(ref_key)
                uni.append(ref_key)
            except KeyError:
                pass
        dic = dict(zip(uni,list(range(10,len(uni) + 10))))
        batch_list = [dic[v] for v in batch_val]
        if ref_key is None:
            ref_batch = None
        else:
            ref_batch = dic[ref_key]
        temp = combat_handler.combat(data.copy(),batch_list,par_prior=parametric,ref_batch=ref_batch)
        # batch should be given as a list of int
        return temp


class CombatOld:
    def __init__(self):
        pass

    def combat(self, data, batch_list, par_prior, ref_batch):
        """ run old combat """
        return pycombat(data, batch_list, par_prior=par_prior, ref_batch=ref_batch)

class CombatNew:
    def __init__(self):
        pass

    def combat(self, data, batch_list, par_prior, ref_batch):
        """ run new combat """
        mode = "p" if par_prior else "np"
        dat = Combat(mode=mode)
        return dat.fit_transform(data.T, np.array(batch_list))

# 230114
# pycombat version check
try:
    from combat.pycombat import pycombat
    combat_handler = CombatOld()
except ModuleNotFoundError:
    from pycombat.pycombat import Combat
    combat_handler = CombatNew()