from tensorflow.keras.layers import (Input, Add, Dense, Activation,
                                     ZeroPadding2D, BatchNormalization,
                                     Flatten, Conv2D, AveragePooling2D,
                                     MaxPooling2D,GlobalAveragePooling2D)

from tensorflow.keras.models import Model
# import tensorflow as tf
from tensorflow.keras.initializers import glorot_uniform , he_normal
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.utils import set_random_seed # Not in my tensorflow version.


#%%
def CMUDeepLens_resnet_block(X_input, f, filters, stage, block,
                        preactivated = False, downsample = False):

    # Defining name basis
    name_base = "resnet_" + str(stage) + block + '_'

    F1, F2, F3 = filters

    if preactivated:
        X_in = X_input
    else:
        X_in = BatchNormalization(axis = 3, name = name_base + 'add_bn')(X_input)
        X_in = Activation('elu',name=name_base+'_add_actv')(X_in)

    if X_input.shape[-1] != F3 and not downsample:
        print("increase_dims == True",X_input.shape, filters)
     
    stride = 2 if downsample else 1
    X = Conv2D(filters=F1, kernel_size=1, strides=stride, 
               padding='same', name= name_base + 'conv1',
               kernel_initializer=he_normal(), use_bias = False)(X_in)
    X = BatchNormalization(axis=3, name= name_base + 'bn1')(X)
    X = Activation('elu', name= name_base + 'actv1')(X)

    X = Conv2D(filters=F2, kernel_size=f, strides=1,
               padding='same', name=name_base + 'conv2',
               kernel_initializer=he_normal(), use_bias = False)(X)
    X = BatchNormalization(axis=3, name=name_base + 'bn2')(X)
    X = Activation('elu', name=name_base+'actv2')(X)


    X = Conv2D(filters=F3, kernel_size=1, strides=1,
               padding='same', name=name_base + 'conv3',
               kernel_initializer= he_normal())(X)
    X = Activation('linear', name=name_base+'actv3')(X)

    if downsample:
        X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=stride,
                        padding='same', name=name_base + 'shortcut_conv',
                        kernel_initializer=he_normal())(X_in)
    else:
        X_shortcut = X_input

    X = Add(name=name_base + 'add')([X, X_shortcut])

    return X

#%%
def DeepLens(input_shape = (45, 45, 1), classes = 2, seed = None):
#    if seed is not None:
#        set_random_seed(seed)

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)
    
    # Stage 1
    # The Theano original implementation had sqrt(12/(in+out)) for the range of the initialization distribution, here is not 12 but 6.
    X = Conv2D(32, 7, strides = 1, name = 'conv1', padding='same',
                   kernel_initializer = glorot_uniform(), use_bias = False)(X_input)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('elu',name='actv_conv1')(X)
    # X = MaxPooling2D((3, 3), strides=(2, 2))(X) # Turns out, CMUDeepLens does not use a MaxPool at the begining.

    # Stage 2
    X = CMUDeepLens_resnet_block(X, 3, [16, 16, 32], stage=2, block='a', preactivated = True)
    X = CMUDeepLens_resnet_block(X, 3, [16, 16, 32], stage=2, block='b')
    X = CMUDeepLens_resnet_block(X, 3, [16, 16, 32], stage=2, block='c')

    # Stage 3 
    X = CMUDeepLens_resnet_block(X, f=3, filters=[32, 32, 64],
                            stage=3, block='a', downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, [32, 32, 64], stage=3, block='b')
    X = CMUDeepLens_resnet_block(X, 3, [32, 32, 64], stage=3, block='c')

    # Stage 4
    X = CMUDeepLens_resnet_block(X, f=3, filters=[64, 64, 128],
                            stage=4, block='a', downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, [64, 64, 128], stage=4, block='b')
    X = CMUDeepLens_resnet_block(X, 3, [64, 64, 128], stage=4, block='c')
    
    # Stage 5
    filters = [128, 128, 256]
    X = CMUDeepLens_resnet_block(X, f=3, filters=filters,
                            stage=5, block='a', downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=5, block='b')
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=5, block='c')
    
    # Stage 6
    filters = [256, 256, 512]
    X = CMUDeepLens_resnet_block(X, f=3, filters=filters,
                            stage=6, block='a',  downsample=True)
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=6, block='b')
    X = CMUDeepLens_resnet_block(X, 3, filters, stage=6, block='c')

    # AVGPOOL
    # X = AveragePooling2D(pool_size=(2,2), padding='same')(X)
    X = GlobalAveragePooling2D(data_format='channels_last', name="globalAveragePool")(X)
    # Output layer
    # X = Flatten()(X) #Already done by the GlobalAveragePooling2D

    X = Dense(1, name='fc' + str(classes), 
              kernel_initializer = glorot_uniform())(X) 
    X = Activation('sigmoid',name='actv_fc'+str(classes))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='CMUDeepLens')

    return model
