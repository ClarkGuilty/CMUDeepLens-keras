#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:13:14 2022

@author: Javier Alejandro Acevedo Barroso
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.CMUDeepLens.deeplens import DeepLens
from src.CMUDeepLens.utils import load_data

from numpy.random import default_rng
from sklearn.model_selection import train_test_split

from astropy.table import Table
from os.path import join

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys


K.set_image_data_format('channels_last')

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%
if len(sys.argv) > 1:
    print("random seed:",sys.argv[1])
    random_seed=int(sys.argv[1])
else:
    random_seed=1

#%%
X_train, X_test, y_train, y_test, prevalence = load_data(random_seed=random_seed)

#%%
model = DeepLens(input_shape = (44, 44, 1), classes = 2)

epochs=100
epochs_drop = epochs//4
# epochs_drop = 40
def scheduler(epoch):
   initial_lrate = 0.001
   drop = 0.1
   lrate = initial_lrate * np.power(drop,np.floor((epoch)/epochs_drop))
   return lrate


callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

model.compile(optimizer=Adam(),
               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy',
                       metrics.Precision(thresholds=0.9),
                       metrics.Recall(thresholds=0.9)]
              )

#%%

train_generator = ImageDataGenerator(rotation_range=0,
                             zoom_range=0,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='wrap',
                             data_format="channels_last")
# fits the model on batches with real-time data augmentation:

#%%
history = model.fit(train_generator.flow(X_train,y_train,256),
          validation_data=(X_test,y_test),verbose=2,
          validation_freq=10, callbacks=[callback],
          epochs = 2)

# %%

model.save("model" + str(random_seed))

preds = model.evaluate(X_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print (preds)

