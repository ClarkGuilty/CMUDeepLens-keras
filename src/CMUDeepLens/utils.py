from os.path import join
from astropy.table import Table

import numpy as np


def int_round(x):
    return int(np.round(x))

#Based on resample_for_bootstrap by Elodie.
def train_test_split_with_prevalences(x,y,test_size=None, random_seed = None,
                                      shuffle = True, proportion = None):
    
    if test_size == None:
        test_size = 1-0.7043 #Around 15000 in training set, depends on the proportion.
    
    rng = np.random.default_rng(seed=random_seed)
    
    if proportion == None:
        proportion = rng.uniform(0.2, 0.4,)
        # proportion = 0.5
    
    
    no_of_items = y.shape[0]
    
    print("Prevalence of lenses:", proportion, ", seed:",random_seed)
    
    new_no_lenses = int_round(no_of_items * proportion * (1-test_size))
    new_no_nonlenses = int_round(new_no_lenses / proportion - new_no_lenses)
    
    lens_ii = rng.choice(np.nonzero(y.reshape(-1))[0] , new_no_lenses, replace=False)
    nonlens_ii = rng.choice(np.nonzero(y.reshape(-1) == 0)[0] , new_no_nonlenses, replace=False)
    train_ii = np.concatenate((lens_ii,nonlens_ii))
    rng.shuffle(train_ii)
    test_ii = np.setdiff1d(np.arange(len(y)), train_ii,assume_unique=False)
    rng.shuffle(test_ii)
    return x[train_ii], x[test_ii], y[train_ii], y[test_ii], proportion
    
#%%
# test_size=1-0.7043
# # test_size=0.3
# # print( train_test_split_with_prevalences(X,y, random_seed=1, test_size=test_size))
# X_train, X_test, y_train, y_test, prevalence = train_test_split_with_prevalences(X,y, random_seed=1, test_size=test_size)
# rng = np.random.default_rng(seed=1)
#%%

# lens_ii = rng.choice(np.nonzero(y.reshape(-1))[0] , 6000,replace=False)
# nonlens_ii = rng.choice(np.nonzero(y.reshape(-1) == 0)[0] , 9000, replace=False)
# train_ii = np.concatenate((lens_ii,nonlens_ii))
# len(train_ii)
# rng.shuffle(train_ii)
# len(train_ii)
# len(np.unique(train_ii))
# print(new_no_lenses, new_no_nonlenses)
# print(len(train_ii))
# test_ii = np.setdiff1d(np.arange(len(y)), train_ii,assume_unique=False)
# len(np.setdiff1d(np.arange(len(y)),t))

#%%

def load_data(data_path="../CMUDeepLensOnUsedData/Data", 
              data_file='CFIS_training_data.hdf5',
              prevalence = None, test_size = None,
              random_seed = 1):

    # Loads the table created in the previous section
    d = Table.read(join(data_path,'CFIS_training_data.hdf5')) #Data Elodie used to train the original network.

    size = 44

    X = np.array(d['image']).reshape((-1,size,size,1))
    y = np.array(d['classification']).reshape((-1,1))

    X_train, X_test, y_train, y_test, prevalence = train_test_split_with_prevalences(
        X,y, random_seed=random_seed, test_size=test_size, proportion=prevalence)

    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test) #The example uses kind of a MinMax scaling. TODO: to try that.

    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))
    return X_train, X_test, y_train, y_test, prevalence