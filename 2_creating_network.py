#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 01:16:30 2022

@author: Javier Alejandro Acevedo Barroso
Very heavily inspired by
https://github.com/marcopeix/Deep_Learning_AI/blob/master/4.Convolutional%20Neural%20Networks/2.Deep%20Convolutional%20Models/Residual%20Networks.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.utils import layer_utils
# from tensorflow.keras.utils.data_utils import get_file
from tensorflow.keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.utils.vis_utils import model_to_dot
# from tensorflow.keras.utils import plot_model
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
# import scipy.misc
from tensorflow.keras.losses import BinaryCrossentropy

from astropy.table import Table
from astropy.io import fits
from os.path import join

import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
#%%

def identity_block(X, f, filters, stage, block):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('elu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)
    
    return X

#%%
def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), 
               padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('elu')(X)


    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)

    
    return X

#%%
def resnet_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), 
               padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('elu')(X)


    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('elu')(X)

    
    return X
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def DeepLens(input_shape = (44, 44, 1), classes = 2):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    # The Theano original implementation had sqrt(12/(in+out)), here is not 12 but 6.
    X = Conv2D(32, (7, 7), strides = (2, 2), name = 'conv1',
               kernel_initializer = glorot_uniform(seed=0))(X) 
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('elu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = identity_block(X, 3, [16, 16, 32], stage=2, block='a')
    X = identity_block(X, 3, [16, 16, 32], stage=2, block='b')
    X = identity_block(X, 3, [16, 16, 32], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f=3, filters=[32, 32, 64],
                            stage=3, block='a', s=2)
    X = identity_block(X, 3, [32, 32, 64], stage=3, block='b')
    X = identity_block(X, 3, [32, 32, 64], stage=3, block='c')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[64, 64, 128],
                            stage=4, block='a', s=2)
    X = identity_block(X, 4, [64, 64, 128], stage=4, block='b')
    X = identity_block(X, 4, [64, 64, 128], stage=4, block='c')
    
    # Stage 5
    filters = [128, 128, 256]
    X = convolutional_block(X, f=3, filters=filters,
                            stage=5, block='a', s=2)
    X = identity_block(X, 4, filters, stage=5, block='b')
    X = identity_block(X, 4, filters, stage=5, block='c')
    
    # Stage 6
    filters = [256, 256, 512]
    X = convolutional_block(X, f=3, filters=filters,
                            stage=6, block='a', s=2)
    X = identity_block(X, 4, filters, stage=6, block='b')
    X = identity_block(X, 4, filters, stage=6, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='DeepLens')

    return model

#%%
from numpy.random import default_rng
from sklearn.model_selection import train_test_split

export_path="/home/javier/myGitStuff/CMUDeepLensOnUsedData/Data"
# Loads the table created in the previous section
d = Table.read(join(export_path,'CFIS_training_data.hdf5')) #Data Elodie used to train the original network.

size = 44

rng = default_rng()
numbers = rng.choice(len(d), size=len(d), replace=False)

X = np.array(d['image'])[numbers].reshape((-1,size,size,1))
y = np.array(d['classification'])[numbers].reshape((-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.9)

X_train = (X_train - np.mean(X_train)) / np.std(X_train)
X_val = (X_val - np.mean(X_val)) / np.std(X_val)
X_test = (X_test - np.mean(X_test)) / np.std(X_test) #The example uses kind of a MinMax scaling. TODO: to try that.

print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(y_test.shape))

#%%
from tensorflow.nn import weighted_cross_entropy_with_logits 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
model = DeepLens(input_shape = (44, 44, 1), classes = 1)

epochs=20
epochs_drop = epochs//4
# epochs_drop = 40
def scheduler(epoch):
   initial_lrate = 0.001
   drop = 0.1
   lrate = initial_lrate * np.power(drop,np.floor((epoch)/epochs_drop))
   return lrate
#%%

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
# model.fit(datagen.flow(x_train, y_train, batch_size=32),
#           steps_per_epoch=len(x_train) / 32, epochs=100)
#%%
# history = model.fit(x=X_train, y=y_train, batch_size = 256,
history = model.fit(train_generator.flow(X_train,y_train,256),
          validation_data=(X_val,y_val),verbose=2,
          validation_freq=10, callbacks=[callback],
          epochs = epochs)

#%%
preds = model.evaluate(X_test, y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
