# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:14:45 2019

@author: tadahaya
"""
import pandas as pd
import numpy as np

from . import processor as pr
from .id_handler.converter import simple_converter
from .batch_handler import Batch
from .quality_checker import Checker


class Profile():
    """
    a class for handling transcriptome profile data

    << DSEIGN >>
    0. preparation (index, columns, batch)
    1. imputation & trim
    2. take logarithm 2
    3. ID conversion
    4. index summarization (n > 2:median, n <= 2:mean)
    5. batch correction
    6. quantile normalization
    7. calculate difference
    8. quality check
    9. consensus signature
    10. sample summarization (mean)
    
    """
    def __init__(self):
        self.converter = None
        self.batch = Batch()
        self.checker = Checker()
        self.data = pd.DataFrame()
        self.temp = pd.DataFrame()


    def get_data(self,convert:bool=True,key:str="",with_id:bool=True,sep:str="_"):
        """
        export data
        
        Parameters
        ----------
        convert: bool
            whether the columns of the output is converted to data of the indicated key

        key: str
            indicates the key for samples

        with_id: bool
            whether ID is added to the tail of sample names

        sep: str
            indicate the separator for bridging the original names and batch values
        
        """
        if convert:
            converter = self.sample_converter(key,with_id,sep)
            temp = self.data.copy()
            col = list(temp.columns)
            temp.columns = [converter[v] for v in col]
            return temp
        else:
            return self.data


    # 0. preparation (index, columns, batch)
    def set_batch(self,data:pd.DataFrame):
        """
        set batch information

        Parameters
        ----------
        data: dataframe
            batch information (sample x batch)
            index should be primary ID for data
        
        """
        self.batch.set_data(data)


    def set_data(self,data:pd.DataFrame,**kwargs):
        """
        set data

        Parameters
        ----------
        data: dataframe
            data to be processed (feature x sample)

        show: bool
            whether distribution is visualized for check
        
        """
        if self.batch.data.shape[0]==0:
            raise ValueError('!! No batch: set_batch before this process !!')
        self.temp = data
        self.update()
        self.compare_correspondence()
        

    # 1. imputation & trim
    def imputer(self,analog:str="",cutoff:float=1.0,rate:float=0.8,update:bool=True,show:bool=True,**kwargs):
        """
        imputing nan and trim the values less than the indicated value
        combination of two imputation methods:
            1. replicate imputation
            2. whole imputation
        
        Parameters
        ----------
        df: a dataframe
            a dataframe to be analyzed (feature x sample)
        
        analog: str
            indicate the name of column of biologically expected analogues in batch data

        rate: float, default 0.8
            determine the ratio of not nan for reliable genes

        cutoff: float
            indicate the threshold value for trimming

        update: bool
            whether data is replaced with the processed one or not

        show: bool
            whether distribution is visualized for check
            
        Returns
        ----------
        res: a dataframe
        
        """
        if len(analog) > 0:
            self.batch.set_expected_analog(analog)
        else:
            if len(self.batch.expected_analog)==0:
                raise ValueError("!! Indicate the key for biologically expected analogues !!")
        try:
            temp = self.batch.adjust2data(self.data)
            replicate = list(temp[self.batch.expected_analog])
        except KeyError:
            raise KeyError("!! Wrong rep: check rep and indicate the name of column indicating biological replicates in batch data !!")
        self.temp = pr.imputer(self.data,replicate,cutoff,rate)
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="1. imputation & trim",**kwargs)


    # 2. take logarithm 2
    def log2(self,rate:float=0.8,cutoff:float=0.0,forced:bool=False,update:bool=True,show:bool=True,**kwargs):
        """
        take logarithm 2
        
        Parameters
        ----------
        rate: float, default 0.8
            determine the ratio of not nan for reliable genes

        cutoff: float
            indicate the threshold value for trimming

        forced: bool
            whether forcedly take logarithm or not

        update: bool
            whether data is replaced with the processed one or not

        show: bool
            whether distribution is visualized for check

        """
        if np.max(self.data.values) < 500:
            if forced==False:
                raise ValueError("!! Data seems to be log: use 'forced' option to take logarithm !!")
        temp = np.log2(self.data)
        temp = temp.where(temp > cutoff).dropna()
        self.temp = temp
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="2. log2 conversion",**kwargs)


    # 3. ID conversion
    def id2gene(self,ref:pd.DataFrame,key_id:str="",key_symbol:str="symbol",drop:bool=True,
                replace_word:str="NOT_FOUND",update:bool=True,show:bool=True,**kwargs):
        """
        convert probe IDs to gene symbols

        Parameters
        ----------
        ref: dataframe
            a dataframe correspondence table between ID and gene name

        key_id, key_symbol: str
            indicate keys for ID and symbols
            if nothing, index is employed as ID

        drop: bool
            indicate whether not found keys are deleted or not

        replace_word: str
            indicate the word for replacing not found keys

        update: bool
            whether data is replaced with the processed one or not

        show: bool
            whether distribution is visualized for check
            
        """
        if len(key_id) > 0:
            ids = list(ref[key_id])
        else:
            ids = list(ref.index)
        symbol = list(ref[key_symbol])
        dic = dict(zip(ids,symbol))
        idx = list(self.data.index)
        new,not_found = simple_converter(idx,dic,True,replace_word)
        self.temp = self.data.copy()
        self.temp.index = new
        if drop:
            remain = [i for i,v in enumerate(new) if v!=replace_word]
            self.temp = self.temp.iloc[remain,:]
        if len(not_found) > 0:
            print("{} IDs were not identifed".format(len(not_found)))
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="3. ID conversion",**kwargs)


    # 4. index summarization (n > 2:median, n <= 2:mean)
    def summarize_index(self,update:bool=True,show:bool=True,**kwargs):
        """ index summarization """
        self.temp = self.data.groupby(axis=0,level=0).median()
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="4. index summarization",**kwargs)


    # 5. batch correction
    def correct_batch(
            self,exp_batch:list=[],pre_quantile:bool=True,sorting:bool=True,
            parametric:bool=True,update:bool=True,show:bool=True,method:str='median',
            ref_key:str=None,**kwargs
            ):
        """
        batch correction with pycombat
        refer to https://epigenelabs.github.io/pyComBat/

        Paremeters
        ----------
        exp_batch: list
            indicate the name of column of experimental batch in batch data

        pre_quantile: bool
            whether quantile normalization is done before Combat
            this process will be done for each batch

        sorting: bool
            whether batches are sorted by the number of contents
            the order affects the order of batch correction with Combat

        parametric: bool
            indicate whether parametric or non-parametric estimation is employed

        update: bool
            whether data is replaced with the processed one or not

        show: bool
            whether distribution is visualized for check

        method: str, default "median"
            determine median or mean values are employed as the template in pre_quantile

        ref_key: str
            indicatest the batch ID
            whose mean and scale are employed for general mean and scale in integration
            this argument is mainly utilized in CombinedProfile class

        """
        self.temp = self.batch.correct_batch(
            self.data,exp_batch=exp_batch,pre_quantile=pre_quantile,method=method,
            parametric=parametric,sorting=sorting,ref_key=ref_key)
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="5. batch correction",**kwargs)


    # 6. quantile normalization
    def quantile(self,update:bool=True,show:bool=True,**kwargs):
        """ quantile normalization """
        self.temp = pr.quantile(self.data)
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="6. quantile",**kwargs)


    # 7. calculate difference
    def diff(self,key:str="",control:str="",method:str="madz",drop:bool=True,whole_loc:bool=False,
             whole_scale:bool=False,forced:bool=False,update:bool=True,show:bool=True,**kwargs):
        """
        calculate difference from the control

        Parameters
        ----------
        key: str
            the key for indicating the batch column to be searched

        control: list
            indicates the control column names

        method: str
            indicate the method for calculating difference from the control
            z, madz, or robz

        drop: boolean
            whether drop inf and nan

        whole_loc,whole_scale: bool
            whether to use loc and scale parameters derived from whole

        forced: bool
            forced option by using whole loc and scale when the number of control is less than 3

        """
        indicator = self.batch.extract_batch(key,self.data)
        indicator = [str(v) + "_{}".format(i) for i,v in enumerate(indicator)]
        dic = dict(zip(indicator,list(self.data.columns)))
        df = self.data.copy()
        df.columns = indicator
        df = pr.diff(df,control,method,drop,whole_loc,whole_scale,forced)
        col = [dic[v] for v in list(df.columns)]
        df.columns = col
        self.temp = df
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="7. difference calculation",**kwargs)


    # 8. check quality
    def check_quality(self,analog:str="",fold:float=1.5,spearman:bool=True,update:bool=True,show:bool=True,**kwargs):
        """
        check the quality of data based on biological replicate similarity
        if correlation between a sample and its replicates are statistically high, the sample is qualified
        
        Parameters
        ----------
        analog: str
            indicate the name of column of biologically expected analogues in batch data

        fold: float
            indicate threshold determining the outliers

        spearman: bool
            whether Spearman's or Pearson's correlation
            
        """
        if len(analog) > 0:
            self.batch.set_expected_analog(analog)
        else:
            if len(self.batch.expected_analog)==0:
                raise ValueError("!! Indicate the key for biologically expected analogues !!")
        if spearman:
            data = self.data.rank()
        else:
            data = self.data
        corr = data.corr()
        outliers = self.checker.corr_outlier(corr,fold)
        del corr
        replicate = self.batch.extract_batch(key=analog,data=self.data)
        col = list(self.data.columns)
        batch = dict(zip(col,replicate))
        qualified = set()
        for k,v in outliers.items():
            if len(v) > 0:
                temp = batch[k]
                test = {w for w in v if temp==batch[w]} | {k}
                if len(test) > 1:
                    qualified = qualified | test
        col2 = [v for v in col if v in qualified]
        df = self.data[col2]
        self.temp = df
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="8. quality check",**kwargs)


    # 9. consensus signature
    def consensus_sig(self,analog:str,sep:str="///",update:bool=True,show:bool=True,**kwargs):
        """
        obtain consensus signature
        
        Parameters
        ----------
        analog: str
            indicate the name of column of biologically expected analogues in batch data

        sep:
            indicate the separator for adding the unique keys
        
        """
        indicator = self.batch.extract_batch(analog,self.data)
        indicator = [str(v) + "{0}{1}".format(sep,i) for i,v in enumerate(indicator)]
        df = self.data.copy()
        df.columns = indicator
        dic = dict(zip(indicator,list(self.data.columns)))
        df = pr.consensus_sig(df,sep=sep,position=0)
        col = [dic[v] for v in list(df.columns)]
        df.columns = col
        self.temp = df
        if update:
            self.update()
        if show:
            self.checker.distribution(self.temp,title="9. consensus signature",**kwargs)


    # 10. summarize sample
    def summarize_sample(self,analog:str,key:str="",with_id:bool=True,sep:str="_",show:bool=True,**kwargs):
        """
        summarize samples with mean
        
        Parameters
        ----------
        analog: str
            indicate the name of column of biologically expected analogues in batch data

        key: str
            indicates the key for samples

        with_id: bool
            whether ID is added to the tail of sample names

        sep: str
            indicate the separator for bridging the original names and batch values

        """
        if len(analog) > 0:
            self.batch.set_expected_analog(analog)
        else:
            if len(self.batch.expected_analog)==0:
                raise ValueError("!! Indicate the key for biologically expected analogues !!")
        converter = self.sample_converter(key,with_id,sep)
        analog_id = self.batch.extract_batch(analog,self.data)
        col = list(self.data.columns)
        dic = dict(zip(analog_id,col))
        df = self.data.copy()
        df.columns = analog_id
        df = df.groupby(level=0,axis=1).mean()
        col = [converter[dic[v]] for v in df.columns]
        df.columns = col
        self.data = df
        if show:
            self.checker.distribution(self.temp,title="10. sample summarization",**kwargs)
        return self.data


    # utilities
    def sample_converter(self,key:str="",with_id:bool=True,sep:str="_"):
        """ convert sample IDs to names """
        if key not in list(self.batch.data.columns):
            raise ValueError("!! No such key in batch information: indicate the correct key for sample name conversion !!")
        else:
            temp_name = list(self.batch.data[key])            
            temp_id = self.batch.get_id()
            if with_id:
                comb = [str(k) + sep + str(i) for k,i in zip(temp_name,temp_id)]
            else:
                comb = temp_name
            dic = dict(zip(temp_id,comb))
            return dic


    def check_dist(self,fixed:list=[],resampling:bool=False,n:int=5,**kwargs):
        """
        check distribution of randomly sampling data

        Parameters
        ----------
        fixed: list
            a list of keys to be visualized

        resampling: bool
            resampling samples or the same samples

        n: int
            sample size

        """
        if self.data.shape[0]==0:
            raise ValueError('!! No data: use set_data before this process !!')
        if resampling:
            fixed = self.checker.sampling(self.data,n)
        self.checker.distribution(self.data,fixed,n,**kwargs)


    def update(self):
        """
        update data by replacing temp and data
        
        Parameters
        ----------
        key: list
            indicates the keys for data to be updated 
        
        """
        if self.temp.shape[0]==0:
            raise ValueError("!! temp is empty: conduct some processing before update the stored data with the processed one !!")
        self.data = self.temp.copy()
        self.checker.update_sample(list(self.data.columns))


    def compare_correspondence(self):
        """ check whether data and batch are correspondent with each other """ 
        data_label = list(self.data.columns)
        batch_label = self.batch.ID
        posi = []
        nega = []
        for d in data_label:
            if d in batch_label:
                posi.append(d)
            else:
                nega.append(d)
        n = len(nega)
        if n > 0:
            print('the following {} samples were not found in the batch and excluded:'.format(n))
            for i in nega:
                print(i)
            self.data = self.data[posi]


    def extract_data(self,key:str="",match:list=[],include:list=[],replace:bool=True):
        """
        extract data based on the indicated batch information

        Parameters
        ----------
        key: str
            indicate the key for extraction

        match, include: list
            indicate keywords for selecting data
            match: whetehr perfectly match
            include: whether include the keyword

        replace: bool
            whether to replace stored data with the extracted one

        """
        self.temp = self.batch.extract_data(self.data,key,match,include)
        if replace:
            self.update()
        return self.temp

 
    def extract_batch(self,key:str=""):
        """
        extract batch data indicated by the given key

        Parameters
        ----------
        key: str
            indicate the key for extraction

        """
        return self.batch.extract_batch(key,self.data)


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
        self.batch.add_expected_analog(key,prefix,sep)


class CombinedProfile():
    """ combine profiles for data integration based on batch correction """
    def __init__(self):
        self.profiles = dict()
        self.keys = []


    def profile(self,key:str=None):
        """
        generate a Profile object

        Parameters
        ----------
        key: str
            the key for the new object
        
        """
        if key is None:
            key = 'key0'
            if len(self.keys)!=0:
                count = 1
                while key in self.keys:
                    key = 'key{}'.format(count)
                    count += 1
        elif key in self.keys:
            raise KeyError('!! the given key is already in the registered keys: change the key !!')
        self.keys.append(key)
        self.profiles[key] = Profile()


    def add_batch(self,data:pd.DataFrame,key:str=None):
        """
        set batch information

        Parameters
        ----------
        data: dataframe
            batch information (sample x batch)
            index should be primary ID for data

        key: str
            indicates the key for the given data
        
        """
        try:
            self.profiles[key].set_batch(data)
        except KeyError:
            raise KeyError('!! Wrong key: check keys attribute or prepare new profile !!')


    def add_data(self,data:pd.DataFrame,key:str=None,**kwargs):
        """
        set data

        Parameters
        ----------
        data: dataframe
            data to be processed (feature x sample)

        key: str
            indicates the key for the given data

        show: bool
            whether distribution is visualized for check
        
        """
        try:
            self.profiles[key].set_data(data,**kwargs)
        except KeyError:
            raise KeyError('!! Wrong key: check keys attribute or prepare new profile !!')        


    def combine(
        self,keys:list=[],ref_key:str=None,pre_quantile:bool=True,
        method:str='median',parametric:bool=True,sep:str='///'
        ):
        """
        combine the indicated profile data sets
        returns dict of batch corrected data sets between them

        Parameters
        ----------
        keys: list
            indicates the profile data sets to be combined
            all data sets are united in the default setting
        
        ref_key: str
            indicatest the key of a data set
            whose mean and scale are employed for general mean and scale in integration

        pre_quantile: bool
            whether quantile normalization is done before Combat
            this process will be done for each batch

        method: str, default "median"
            determine median or mean values are employed as the template in pre_quantile

        parametric: bool
            indicate whether parametric or non-parametric estimation is employed

        sep:
            indicate the separator for adding the unique keys
            sep should not be included in the original IDs

        """
        if len(keys)==0:
            keys = self.keys
        datasets = []
        batches = []
        for k in keys:
            pro = self.profiles[k]
            data = pro.data
            batch = pro.batch
            b = batch.adjust2data(data)
            b.loc[:,'_COMBINED'] = [k] * b.shape[0]
            idx = [str(v) + sep + k for v in list(b.index)]
            b.index = idx
            data.columns = idx
            datasets.append(data)
            batches.append(b)
        datasets = pd.concat(datasets,join='inner',axis=1)
        batches = pd.concat(batches,join='inner',axis=0)
        dat = Profile()
        dat.set_batch(batches)
        dat.set_data(datasets)
        dat.correct_batch(
            exp_batch=['_COMBINED'],pre_quantile=pre_quantile,method=method,
            parametric=parametric,sorting=False,ref_key=ref_key
            )
        res = dat.data
        idx = [v.split('///')[-1] for v in list(res.columns)]
        res_dic = dict()
        for k in keys:
            mask = [True if v==k else False for v in idx]
            temp = res.loc[:,mask]
            temp_col = list(temp.columns)
            temp_col = [c.split(sep)[0] for c in temp_col]
            temp.columns = temp_col
            res_dic[k] = temp
        return res_dic