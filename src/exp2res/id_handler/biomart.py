# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:05:34 2019

@author: tadahaya
"""
import pandas as pd
import os

from .converter import SynoDict

if os.name == "nt":
    DIR_SEP = "\\"
elif os.name == 'posix':
    DIR_SEP = "/"

class Biomart():
    """ a class for ID data handler """
    def __init__(self):
        self.__fileprefix = "mart_export_" # hard coding
        self.__sp_dict = {"human":"HGNC", # hard coding
                          "mouse":"MGI", # hard coding
                          "rat":"RGD"} # hard coding
        self.dic = SynoDict()
        self.__base = os.path.dirname(__file__) # ~\id_handler
        self.__state = {"database":"BioMart","name":""}


    def get_dict(self):
        return self.dic


    def set_data(self,species="human"):
        """ get converter pickle path derived from biomart """
        print("species='{}'".format(species))
        key = self.__fileprefix + self.__sp_dict[species]
        p_pkl = self.__checker(self.__base,key,extension="pkl")
        if len(p_pkl)==0:
            p_txt = self.__checker(self.__base,key,extension="txt")
            if len(p_txt)==0:
                raise ValueError("!! check data existence !!")
            else:
                self.__generate_conv(p_txt,self.__sp_dict[species])
        else:
            self.dic.read_pickle(p_pkl)
        self.__state["name"] = species


    def prep_(self):
        raise NotImplementedError


    def get_state(self):
        return self.__state


    def __checker(self,url,key,extension="pkl"):
        """ check existence of files with the indicated extension """
        f_intr = "{0}.{1}".format(key,extension)
        p_intr = url + DIR_SEP + "{}".format(f_intr)
        ps = os.listdir(url)
        if f_intr in ps:
            return p_intr
        else:
            return ""


    def __generate_conv(self,url,key):
        """ generate dictionary from Biomart data """
        names,ids,synonyms = self.__prep_gene_conv(url,key + " symbol")
        print("--- initial dictionary prepration (take a long time) ---")
        self.dic = SynoDict(names,ids,synonyms,processing=True)
        self.dic.to_pickle(url.replace("txt","pkl"))


    def __prep_gene_conv(self,url,symbol_name="HGNC symbol"): # hard coding
        """
        generate necessary information for converter from BioMart data
        HGNC/MGI/RGD symbol, NCBI gene ID, Gene synonym
        
        Parameters
        ----------
        url: str
            a path for the BioMart derived file

        symbol_name: str
            indicate the column name for symbol
            "HGNC symbol": human
            "MGI symbol": mouse
            "RGD symbol": rat

        """
        GENEID = "NCBI gene ID" # hard coding
        SYNONYM = "Gene Synonym" # hard coding
        df = pd.read_csv(url,sep="\t").dropna(subset=[GENEID])
        df.loc[df[SYNONYM]!=df[SYNONYM],SYNONYM] = df[symbol_name]
        df = df.groupby([GENEID],as_index=False).agg(set)
        df[GENEID] = df[GENEID].astype(int)
        df[symbol_name] = df[symbol_name].apply(lambda x: x.pop())
        df = df.dropna(subset=[symbol_name,GENEID])
        df.set_index(symbol_name,drop=True,inplace=True)
        return [v.lower() for v in list(df.index)],list(df[GENEID]),list(df[SYNONYM])