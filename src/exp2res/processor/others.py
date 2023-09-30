# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 10:39:56 2018

other utility modules
version: 210111

@author: tadahaya
"""
import pandas as pd
import numpy as np

__all__ = ["undo_autoconv"]


def undo_autoconv(data):
    """ correct wrong indices by excel """
    DEC = {}
    MARCH = {}
    OCT = {}
    SEPT = {}
    for i in range(20):
        DEC["{}-Dec".format(i)] = "DEC{}".format(i)
        DEC["{}-DEC".format(i)] = "DEC{}".format(i)
        MARCH["{}-Mar".format(i)] = "MARCH{}".format(i)
        MARCH["{}-MAR".format(i)] = "MARCH{}".format(i)
        OCT["{}-Oct".format(i)] = "OCT{}".format(i)
        OCT["{}-OCT".format(i)] = "OCT{}".format(i)
        SEPT["{}-Sep".format(i)] = "SEPT{}".format(i)    
        SEPT["{}-SEP".format(i)] = "SEPT{}".format(i)    
    converter = {}
    converter.update(DEC)
    converter.update(MARCH)
    converter.update(OCT)
    converter.update(SEPT)
    if type(data)==list:
        return [converter[v] if v in converter.keys() else v for v in data]
    else:
        ind = list(data.index)
        new = [converter[v] if v in converter.keys() else v for v in ind]
        data.index = new
        data = data.loc[~data.index.str.contains("---"),:]
        return data