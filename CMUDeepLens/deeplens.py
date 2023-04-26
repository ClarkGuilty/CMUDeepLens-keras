from tensorflow.keras.layers import (Input, Add, Dense, Activation,
                                     ZeroPadding2D, BatchNormalization,
                                     Flatten, Conv2D, AveragePooling2D,
                                     MaxPooling2D,Identity)

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def identity_block(X, f, filters, stage, block, preactivated = False):
    
    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    if preactivated:
        X_activated = Identity(X)
    else:
        X_activated = BatchNormalization(axis = 3, name = bn_name_base + 'shortcut_bn')(X)
        X_activated = Activation('elu')(X_activated)
    
    # Save the input value
    X_shortcut = Identity(X_activated)
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1),
               padding = 'same', name = conv_name_base + '2a', #same or valid is the same for kernel_size=1.
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('elu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1),
               padding = 'same', name = conv_name_base + '2b',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('elu')(X)

    # Third component of main path 
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1),
               padding = 'same', name = conv_name_base + '2c', #same or valid is the same for kernel_size=1.
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = Activation('linear')(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    # X = Activation('elu')(X)
    
    return X

#%%
def convolutional_block(X, f, filters, stage, block, s=2,
                        preactivated = False):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    if preactivated:
        X_activated = Identity(X)
    else:
        X_activated = BatchNormalization(axis = 3, name = bn_name_base + 'shortcut_bn')(X)
        X_activated = Activation('elu')(X_activated)
    
    # Save the input value
    X_shortcut = Identity(X)
                
    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), 
               padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X_activated)
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
               padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('linear')(X)

    ##### SHORTCUT PATH #### 
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s),
                        padding='same', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    # X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    # X = Activation('elu')(X)

    
    return X

#%%
def CMUDeepLens_resnet_block(X, f, filters, stage, block, s=2,
                        preactivated = False, downsample = False):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = Identity(X)
                
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), 
               padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('elu')(X)
    
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('elu')(X)


    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1),
               padding='same', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation('linear')(X)

    if downsample:
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s),
                        padding='same', name=conv_name_base + 'shortcut_conv',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)

    X = Add()([X, X_shortcut])
    X = BatchNormalization(axis = 3, name = bn_name_base + 'shortcut_bn')(X)
    X = Activation('elu')(X)
    
    return X

#%%
def DeepLens(input_shape = (45, 45, 1), classes = 2):
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Stage 1
    # The Theano original implementation had sqrt(12/(in+out)) for the range of the initialization distribution, here is not 12 but 6.
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv1', padding='same',
                   kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('elu')(X)
    # X = MaxPooling2D((3, 3), strides=(2, 2))(X) # Turns out, CMUDeepLens does not use a MaxPool at the begining.

    # Stage 2
    X = CMUDeepLens_resnet_block(X, 3, [16, 16, 32], stage=2, block='a',preactivated=True)
    X = CMUDeepLens_resnet_block(X, 3, [16, 16, 32], stage=2, block='b')
    X = CMUDeepLens_resnet_block(X, 3, [16, 16, 32], stage=2, block='c')

    # Stage 3 -
    X = CMUDeepLens_resnet_block(X, f=3, filters=[32, 32, 64],
                            stage=3, block='a', downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, [32, 32, 64], stage=3, block='b')
    X = CMUDeepLens_resnet_block(X, 3, [32, 32, 64], stage=3, block='c')

    # Stage 4
    X = CMUDeepLens_resnet_block(X, f=3, filters=[64, 64, 128],
                            stage=4, block='a', s=2, downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, [64, 64, 128], stage=4, block='b')
    X = CMUDeepLens_resnet_block(X, 3, [64, 64, 128], stage=4, block='c')
    
    # Stage 5
    filters = [128, 128, 256]
    X = CMUDeepLens_resnet_block(X, f=3, filters=filters,
                            stage=5, block='a', s=2, downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=5, block='b')
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=5, block='c')
    
    # Stage 6
    filters = [256, 256, 512]
    X = CMUDeepLens_resnet_block(X, f=3, filters=filters,
                            stage=6, block='a', s=2, downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=6, block='b')
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=6, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc' + str(classes), 
              kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='DeepLens')

    return model