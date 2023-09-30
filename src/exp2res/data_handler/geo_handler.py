# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:14:45 2019

@author: tadahaya
"""
import pandas as pd
import csv

__all__ = ["prep_series","prep_soft"]


def prep_series(url:str=""):
    """
    process a Series file

    Parameters
    ----------
    url: str
        a path for a series file

    """
    if len(url)==0:
        raise ValueError("!! Give url of Series file of interest !!")
    dat = Series()
    dat.set_data(url)
    data = dat.extract_data()
    _ = dat.extract_info()
    batch = dat.extract_batch()
    print("return: extracted data and batch information")
    return data,batch


def prep_soft(url:str="",loc_id:int=0,loc_symbol:int=1,head:int=10,position:int=0,sep:str=" /// "):
    """
    prepare ID converter from Soft file

    Parameteres
    -----------
    url: str
        a path for a soft file

    loc_id: int
        indicate the column number of probe ID

    loc_symbol: int
        indicate the column number of gene symbol

    head: int
        for heading to check the content of Soft file

    position: int
        indicate the position of gene symbol after separation with sep

    sep: str
        indicate the separator for gene symbol processing
    
    """
    if len(url)==0:
        raise ValueError("!! Give url of Soft file of interest !!")
    dat = Soft()
    dat.set_data(url)
    idx_conv = dat.set_annotation(loc_id,loc_symbol,head,position,sep)
    return idx_conv


class Series():
    """ extract data and batch information from Series data """
    def __init__(self):
        self.raw = pd.DataFrame()
        self.info = pd.DataFrame()
        self.__flag = 0
        self.__idx = []

    # setter
    def set_data(self,url:str=""):
        """
        load data

        Parameters
        ----------
        url: str
            a path for a series file

        """
        lst = []
        ap = lst.append
        with open(url,newline="",encoding="utf-8_sig") as f:
            reader = csv.reader(f,delimiter="\t")
            for row in reader:
                ap(row)
        self.raw = pd.DataFrame(lst)        


    def extract_data(self):
        """ extract data from Series """
        self.__idx = list(self.raw.iloc[:,0])
        self.__flag = self.__idx.index("ID_REF")
        
        ### data part
        dat = self.raw.iloc[self.__flag:-1,:]
        l_col = list(dat.iloc[0,1:])
        l_ind = list(dat.iloc[1:,0])
        dat = dat.iloc[1:,1:]
        dat.columns = l_col
        dat.index = l_ind
        dat = dat.replace('"','')
        dat = dat.astype("float")
        return dat


    def extract_info(self):
        """ extract information from Series """
        flag2 = self.__idx.index(None)
        infor = self.raw.iloc[flag2 + 1:self.__flag - 1,:]
        l_ind2 = list(infor.iloc[:,0])
        infor.index = l_ind2
        self.info = infor.iloc[:,1:]
        return self.info


    def extract_batch(self):
        """ extract information from Series """
        extracted = self.info.loc[(self.info.index.str.contains("!Sample_title"))|
                (self.info.index.str.contains("!Sample_geo_accession"))|
                (self.info.index.str.contains("!Sample_source_name_ch1"))|
                (self.info.index.str.contains("!Sample_organism_ch1"))|
                (self.info.index.str.contains("!Sample_characteristics_ch1")),:]
        unique = []
        ap = unique.append
        for i in range(len(extracted)):
            temp = extracted.iloc[i,:]
            if len(list(set(list(temp))))!=1:
                ap(i)
        extracted = extracted.iloc[unique,:]
        extracted1 = extracted.loc[~extracted.index.str.contains("!Sample_characteristics_ch1"),:]    
        extracted1.index = [v.replace("!Sample","sample") for v in list(extracted1.index)]
        extracted2 = extracted.loc[extracted.index.str.contains("!Sample_characteristics_ch1"),:]
        extracted2.index = ["character{}".format(i) for i,v in enumerate(list(extracted2.index))]
        batch0 = pd.concat([extracted1,extracted2],axis=0,join="inner")
        return batch0.T.set_index("sample_geo_accession")


class Soft():
    """ extract annotation information from Soft data """
    def __init__(self):
        self.raw = pd.DataFrame()


    # setter
    def set_data(self,url:str="",head:int=10):
        """
        load data

        Parameters
        ----------
        url: str
            a path for a soft file

        head: int, default 10
            heading to check which column contains ID or symbol

        """
        lst = []
        ap = lst.append
        with open(url,newline="",encoding="utf-8_sig") as f:
            reader = csv.reader(f,delimiter="\t")
            for row in reader:
                if row[0] != "!platform_table_end":
                    ap(row)
                else:
                    break
        df = pd.DataFrame(lst)
        ind = list(df.iloc[:,0])
        flag = ind.index("!platform_table_begin")
        data = df.iloc[flag + 1:,:]
        if head!=0:
            print(data.iloc[:head,:])
            print(data.iloc[0,:])
        else:
            pass
        self.raw = data


    # getter
    def get_data(self):
        """ get raw soft file """
        return self.raw


    def set_annotation(self,loc_id:int=0,loc_symbol:int=1,head:int=10,position:int=0,sep:str=" /// "):
        """
        prepare annotation file from soft

        Parameteres
        -----------
        loc_id: int
            indicate the column number of probe ID

        loc_symbol: int
            indicate the column number of gene symbol

        head: int
            for heading to check the content of Soft file

        position: int
            indicate the position of gene symbol after separation with sep

        sep: str
            indicate the separator for gene symbol processing
        
        """
        print("*** candidate ID column: {} ***".format(loc_id))
        print(self.raw.iloc[:head,loc_id])
        print("*** candidate symbol column: {} ***".format(loc_symbol))
        print(self.raw.iloc[:head,loc_symbol])
        l_id = list(self.raw.iloc[:,loc_id])
        l_symbol = list(self.raw.iloc[:,loc_symbol])        
        l_symbol = list(map(lambda x: str(x),l_symbol))
        l_symbol2 = []
        ap = l_symbol2.append
        for v in l_symbol:
            spl = v.split(sep)
            if len(spl)==1:
                ap(v)
            else:
                ap(spl[position])
        return pd.DataFrame({"symbol":l_symbol2},index=l_id).iloc[1:,:]