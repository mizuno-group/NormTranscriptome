# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:05:34 2019

Converter

@author: tadahaya
"""

import pandas as pd
import numpy as np
from itertools import chain
from tqdm import trange
import pickle
import collections
from collections import defaultdict

def simple_converter(obj:list=[],dic:dict=dict(),replace:bool=False,replace_word:str="NOT_FOUND"):
    """
    convert a list based on the given dictionary

    Parameters
    ----------
    obj: list
        a list to be converted

    dic: dict
        a dictionary for conversion

    replace: bool
        indicate whether not found data is kept or not

    replace_word: str
        indicate the word for replacing not found keys

    """
    new = []
    ap = new.append
    not_found = []
    ap2 = not_found.append
    if replace:
        for v in obj:
            try:
                ap(dic[v])
            except KeyError:
                ap(replace_word)
                ap2(v)
    else:
        for v in obj:
            try:
                ap(dic[v])
            except KeyError:
                ap(v)
                ap2(v)
    return new,not_found


class SynoDict():
    """
    Dict considering synonyms
    time consuming in dict preparation

    Parameters
    ----------
    keys: list
        representative keys for decoder
        
    synonyms: list
        a list of synonym sets [{},{},...]
        !! nan should be replaced with "" before construction !!
        
    values: list
        list of int

    """
    def __init__(self,keys=[],values=[],synonyms=[],processing=True):
        if processing:
            self.keys = keys
            lkeys = list(map(lambda x: str(x).lower(),keys))
            self.values = list(map(lambda x: int(x),values))
            self.decoder = dict(zip(self.values,lkeys))
            synonyms = [set([str(x).lower() for x in v]) for v in synonyms]
            lst = list(chain.from_iterable(synonyms))
            ol = set(k for k,v in collections.Counter(lst).items() if v > 1) | {""}
            synonyms = [v - ol for v in synonyms]
            self.synonyms = [{v} | w for v,w in zip(self.keys,synonyms)]
        else:
            self.keys = keys
            self.values = list(map(lambda x: int(x),values))
            self.decoder = dict(zip(values,keys))
            self.synonyms = synonyms
        try:
            self.sub = max(np.array(values)) + 1 # int used for substitution when KeyError occurs
        except ValueError:
            self.sub = 0
        self.__zip = list(zip(self.synonyms,self.values))
        self.not_found = dict()


    def enc(self,word):
        """ encoder """
        for v,w in self.__zip:
            if word in v:
                return w
        raise KeyError(word)


    def fix(self,obj):
        """
        return fixed dict for converting the indicate list
        
        Parameters
        ----------
        obj: list
            a list of conversion target

        substitute: str
            a word employed for indicating not found keys        

        """
        value = self.enc_list(obj)
        return dict(zip(obj,value))


    def get_not_found(self):
        """ get not found dictionary """
        return self.not_found


    def to_pickle(self,url):
        """ to save Synonym Dict """
        with open(url,"wb") as f:
            pickle.dump([self.keys,self.values,self.synonyms],f)


    def read_pickle(self,url):
        """ to load Synonym Dict """
        with open(url,"rb") as f:
            temp = pickle.load(f)
        self.keys = temp[0]
        self.values = temp[1]
        self.synonyms = temp[2]
        self.decoder = dict(zip(self.values,self.keys))
        self.__zip = list(zip(self.synonyms,self.values))
        self.not_found = dict()
        try:
            self.sub = max(np.array(temp[1])) + 1 # int used for substitution when KeyError occurs
        except ValueError:
            self.sub = 0


    def enc_list(self,target):
        """
        convert a list according to pre-defined dict

        Parameters
        ----------
        target: list
        
        """
        res = []
        ap = res.append
        nf = []
        ap2 = nf.append
        i = 0
        for v in target:
            try:
                ap(self.enc(v))
            except KeyError:
                temp = self.sub + i
                ap(temp)
                self.not_found[v] = temp
                i += 1
        self.keys += list(self.not_found.keys())
        self.synonyms += [set() for l in range(i)]
        self.values += list(self.not_found.values())
        tdic = dict(zip(list(self.not_found.values()),list(self.not_found.keys())))
        self.decoder.update(tdic)
        self.sub += i
        return res


    def dec_list(self,target):
        """ decoder for list """
        return [self.decoder[v] for v in target]


    def enc_set(self,target):
        """
        convert a set according to pre-defined dict

        Parameters
        ----------
        target: set

        substitute: str
            a word employed for indicating not found keys        
        
        """
        res = set()
        ad = res.add
        i = 0
        for v in target:
            try:
                ad(self.enc(v))
            except KeyError:
                temp = self.sub + i
                ad(temp)
                self.not_found[v] = temp
                i += 1
        self.keys += list(self.not_found.keys())
        self.synonyms += [set() for l in range(i)]
        self.values += list(self.not_found.values())
        tdic = dict(zip(list(self.not_found.values()),list(self.not_found.keys())))
        self.decoder.update(tdic)
        self.sub += i
        return res


    def dec_set(self,target):
        """ decoder for set """
        return {self.decoder[v] for v in target}


    def aggregate(self):
        """ aggregate values """
        dup = [x for x in set(self.values) if self.values.count(x) > 1]
        n = len(self.keys)
        if len(dup) > 0:
            new_keys = []
            new_syno = []
            new_idx = []
            ap = new_keys.append
            ap2 = new_syno.append
            for d in dup:
                idx = [i for i,v in enumerate(self.values) if v==d]
                syno = set(chain.from_iterable([self.synonyms[i] for i in idx]))
                n_keys = [len(self.keys[i]) for i in idx]
                ap(self.keys[np.argmin(n_keys)])
                new_idx += idx
            rem_vals = [v for v in self.values if v not in dup]
            rem_keys = [self.keys[i] for i in range(n) if i not in new_idx]
            rem_syno = [self.synonyms[i] for i in range(n) if i not in new_idx]
            self.keys = rem_keys + new_keys
            self.values = rem_vals + dup
            self.synonyms = rem_syno + new_syno


class FixedDict(SynoDict):
    """ handling conversion between names and IDs """
    def __init__(self,keys=[],values=[],synonyms=[]):
        self.keys = keys
        self.values = values
        self.decoder = dict(zip(values,keys))
        self.synonyms = synonyms
        self.sub = max(values) + 1 # int used for substitution when KeyError occurs
        self.not_found = dict()
        self.encoder = dict(zip(self.keys,self.values))


    def enc_list(self,target):
        """
        convert a list according to pre-defined dict

        Parameters
        ----------
        target: list
        
        """
        return [self.encoder[v] for v in target]


    def dec_list(self,target):
        """ decoder for list """
        return [self.decoder[v] for v in target]


    def enc_set(self,target):
        """
        convert a set according to pre-defined dict

        Parameters
        ----------
        target: set
        
        """
        return {self.encoder[v] for v in target}


    def dec_set(self,target,substitute=0):
        """ decoder for set """
        return {self.decoder[v] for v in target}


class Integrator():
    def __init__(self):
        self.ref = None
        self.encoder = dict() # integrated dict for encoding
        self.decoders = dict() # dict of dict for each decoding
        self.keys = []


    def refresh_keys(self):
        """ refresh keys """
        self.keys = []


    def make_ref(self,keys=[],values=[],synonyms=[]):
        """ prepare SynoDict """
        self.ref = SynoDict(keys,values,synonyms,processing=True)
        return self.ref


    def load_ref(self,ref):
        """
        load a reference

        Parameters
        ----------
        ref: SynoDict

        """
        self.ref = ref


    def register(self,keys=[],name=None,ref=None,drop=False):
        """
        registration of a member

        Parameters
        ----------
        name: str
            indicates the key for the data

        keys: list
            data for dictionary preparation
            !! Should be lower case !!
                    
        """
        if len(keys)==0:
            raise ValueError("!! Give keys as a list !!")
        if ref is not None:
            self.ref = ref
        if self.ref is None:
            raise ValueError("!! Load reference before this process !!")
        if name is None:
            name = "decoder" + str(len(self.keys))
        if drop:
            values = self.ref.enc_list(keys)
            nf = set(self.ref.get_not_found().keys())
            keys = [k for k in keys if k not in nf]
            values = [v for k,v in zip(keys,values) if k not in nf]
        else:
            values = self.ref.enc_list(keys)
        dic = dict(zip(keys,values))
        self.encoder.update(dic)
        dec = dict(zip(values,keys))
        self.decoders[name] = dec
        self.keys.append(name)


    def extend_encoder(self,dic:dict,name:str="EXTENDED"):
        """
        extend encoder by adding dic

        Parameters
        ----------
        name: str
            indicates the key for the data
        
        """
        self.encoder.update(dic)
        dec = dict(zip(dic.values(),dic.keys()))
        self.decoders[name] = dec
        self.keys.append(name)


    def merge_decoder(self,keys:list=[],new_name:str="MERGED"):
        """
        merge decoders with the indicated keys

        Parameters
        ----------
        keys: list
            indicates the keys to be concatenated

        new_name: str
            indicates the key for the concatenated data
        
        """
        if len(keys) < 2:
            raise ValueError("!! Few keys for concatenation !!")
        temp = self.decoders[keys[0]]
        for k in keys[1:]:
            temp.update(self.decoders[k])
        self.decoders[new_name] = temp


    def enc_list(self,target):
        """
        convert a list according to the integrated encoder

        Parameters
        ----------
        target: list
        
        """
        try:
            res = [self.encoder[v] for v in target]
        except KeyError("!! Some inputs are not found: register(drop=False) whole keys to be analyzed !!"):
            pass
        return res


    def enc_set(self,target):
        """
        convert a list according to the integrated encoder

        Parameters
        ----------
        target: set
        
        """
        try:
            res = {self.encoder[v] for v in target}
        except KeyError("!! Some inputs are not found: register(drop=False) whole keys to be analyzed !!"):
            pass
        return res


    def dec_list(self,target,key):
        """
        convert a list according to the integrated decoder

        Parameters
        ----------
        target: list
        
        """
        try:
            res = [self.decoders[key][v] for v in target]
        except KeyError("!! Some inputs are not found: register(drop=False) whole keys to be analyzed !!"):
            pass
        return res


    def dec_set(self,target,key):
        """
        convert a list according to the integrated decoder

        Parameters
        ----------
        target: set
        
        """
        try:
            res = {self.decoders[key][v] for v in target}
        except KeyError("!! Some inputs are not found: register(drop=False) whole keys to be analyzed !!"):
            pass
        return res