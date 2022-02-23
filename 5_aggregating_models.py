#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 23:13:39 2022

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

from src.CMUDeepLens.deeplens import DeepLens
from src.CMUDeepLens.utils import load_data
from os.path import join

def load_model(model_number,
              data_path="Models",
              models_name_base = "model"):
    
    model = tf.keras.models.load_model(join(data_path,models_name_base)+str(model_number))
    
    return model

load_model(30)
#%%

X_train, X_test, X_val, y_train, y_test, y_val, prevalence = load_data(random_seed=2)

model1 = load_model(1)
#%%
def predict(model, data = X_test):
    # return np.round(model(data),decimals=0)
    return model(data)
    
#%%
df = pd.DataFrame(y_test,columns=["real"])
#%%

for i in range(1,31):
    df['model'+str(i)] = predict(load_model(i))
#%%
model_keys = df.keys()[1:]
data_keys = df.keys()[0]
#%%
df["mean"] 
np.mean(df[model_keys], axis=1)





































