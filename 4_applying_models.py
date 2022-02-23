#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 00:01:59 2022

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from src.CMUDeepLens.deeplens import DeepLens
from src.CMUDeepLens.utils import load_target_data, load_model
from os.path import join


def apply_model(model, X):
    if type(model) == int:
        model = load_model(model)
    return model.predict(X)

#%%
for i in range(0,4):
    
    X, names = load_target_data(data_file='CFIS_real_data_'+str(i)+'.hdf5')
    print("loaded")
    df = pd.DataFrame({"name": names})

    for j in range(1,31):    
        df["model"+str(j)] = apply_model(j,X)

    df.to_csv(join("Classifications","model"+str(i)+".csv"))
#%%
















