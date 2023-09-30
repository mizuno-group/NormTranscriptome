# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:39:56 2021

calculate difference from control
version: 210111

@author: tadahaya
"""
import pandas as pd
import numpy as np

__all__ = ["diff"]


def diff(data,control:list=[],method:str="madz",drop:bool=True,
         whole_loc:bool=False,whole_scale:bool=False,forced:bool=False):
    """
    to calculate difference from the indicated control data
    
    Parameters
    ----------
    data: dataframe
        a dataframe to be analyzed (feature x sample)
    
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
    if method=="madz":
        dat = MadzDifference()
    elif method=="z":
        dat = ZDifference()
    elif method=="robz":
        dat = RobzDifference()
    else:
        raise ValueError("!! Wrong method: choose z, madz, or robz !!")
    dat.set_data(data,control,forced)
    if whole_loc:
        dat.calc_loc(dat.data)
    else:
        dat.calc_loc(dat.control)
    if whole_scale:
        dat.calc_scale(dat.data)
    else:
        dat.calc_scale(dat.control)
    return dat.calc_diff(drop)


# abstract class
class Difference():
    def __init__(self):
        self.loc = np.array([])
        self.scale = np.array([])
        self.control = pd.DataFrame()
        self.data = pd.DataFrame()
        self.idx = []
        self.col = []


    def set_data(self,data,control:list=[],forced:bool=False):
        """
        set dataframe to be analyzed

        Parameters
        ----------
        data: a dataframe
            a dataframe to be analyzed (feature x sample)
        
        control: list
            indicates the control column names

        forced: bool
            forced option by using whole loc and scale when the number of control is less than 3

        """
        col = list(data.columns)
        col_con = []
        ap = col_con.append
        for i,v in enumerate(col):
            if self.__checker(v,control):
                ap(i)
        col_con = list(set(col_con))
        col_con = [col[i] for i in col_con]
        n_con = len(col_con)
        print("No. of control: ",n_con)
        if n_con < 3:
            if forced:
                self.control = data.copy()
                col_tre = list(data.columns)
            else:
                raise ValueError("!! Control samples are not enough: check the term for control indication or use 'forced' option !!")
        else:
            self.control = data.loc[:,col_con]
            col_tre = [v for v in col if v not in col_con]
        self.data = data.loc[:,col_tre]
        self.idx = list(self.data.index)
        self.col = list(self.data.columns)


    def calc_loc(self):
        """ calculate parmeters """
        raise NotImplementedError


    def calc_scale(self):
        """ calculate parmeters """
        raise NotImplementedError


    def calc_diff(self,drop:bool=True):
        """ calculate difference """
        with np.errstate(divide='ignore',invalid='ignore'):
            df = pd.DataFrame((self.data.values - self.loc)/self.scale,index=self.idx,columns=self.col)
        df = df.replace(np.inf,np.max(df.values))
        df = df.replace(-np.inf,np.min(df.values))
        if drop:
            df = df.dropna()
        return df


    def __checker(self,term,check_list):
        """ check whether a term contains a word in the given list """
        temp = list(map(lambda x: x in term,check_list))
        return np.sum(temp) > 0


# concrete class
class ZDifference(Difference):
    def __init__(self):
        super().__init__()


    def calc_loc(self,data):
        """ calculate location parameter """
        self.loc = np.c_[np.mean(data,axis=1)]


    def calc_scale(self,data):
        """ calculate scale parameter """
        self.scale = np.c_[np.std(data,axis=1,ddof=0)]


# concrete class
class MadzDifference(Difference):
    def __init__(self):
        super().__init__()


    def calc_loc(self,data):
        """ calculate location parameter """
        self.loc = np.c_[np.median(data,axis=1)]


    def calc_scale(self,data):
        """ calculate scale parameter """
        if self.loc.shape[0]==0:
            self.calc_loc(data)
        mad = np.c_[np.median(np.abs(data - self.loc),axis=1)]
        self.scale = 1.4826 * mad


# concrete class
class RobzDifference(Difference):
    def __init__(self):
        super().__init__()


    def calc_loc(self,data):
        """ calculate location parameter """
        self.loc = np.c_[np.median(data,axis=1)]


    def calc_scale(self,data):
        """ calculate scale parameter """
        q1 = np.c_[np.percentile(data,25,axis=1)]
        q3 = np.c_[np.percentile(data,75,axis=1)]
        self.scale = 0.7413 * (q3-q1)
