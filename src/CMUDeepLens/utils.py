from os.path import join
from astropy.table import Table

import numpy as np

from sklearn.model_selection import train_test_split


def int_round(x):
    return int(np.round(x))

#Based on resample_for_bootstrap by Elodie.
#The minimum possible proportion is (no_train_samples - no_non_lenses)/no_train_samples.
def train_test_split_with_prevalences(x,y,test_size=None, random_seed = None,
                                      shuffle = True, proportion = None):
    
    if test_size == None:
        test_size = 1-0.7043 #Around 15000 in training set, depends on the proportion.
    
    rng = np.random.default_rng(seed=random_seed)
    no_train_samples = int_round((1-test_size) * len(y))
    min_proportion = (no_train_samples - len(np.nonzero(y==0)[0])) / no_train_samples
    if proportion == None:
        proportion = rng.uniform(min_proportion, 0.4)
    else:
        if proportion < min_proportion:
            raise ValueError("The minimum proportion allowed for that test_size is " + str(min_proportion))
    
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
#The validation size is set to 10%. This is intentionally not a free argument.
def train_test_validation_split_with_prevalences(x,y,test_size=None,
                                      random_seed = None,
                                      shuffle = True, proportion = None):
    
    if test_size == None:
        test_size = 0.2175 #Around 15000 in training set, depends on the proportion.
    
    # rng0 = np.random.default_rng(seed=9999)
    # train_test_split()
    
    x, x_test, y, y_test = train_test_split(x, y, test_size=0.1,random_state=9999)
    
    rng = np.random.default_rng(seed=random_seed)
    no_train_samples = int_round((1-test_size) * len(y))
    min_proportion = (no_train_samples - len(np.nonzero(y==0)[0])) / no_train_samples
    if proportion == None:
        proportion = rng.uniform(min_proportion, 0.4)
    else:
        if proportion < min_proportion:
            raise ValueError("The minimum proportion allowed for that test_size is " + str(min_proportion))
    
    no_of_items = y.shape[0]
    
    print("Prevalence of lenses:", proportion, ", seed:",random_seed)
    
    new_no_lenses = int_round(no_of_items * proportion * (1-test_size))
    new_no_nonlenses = int_round(new_no_lenses / proportion - new_no_lenses)
    
    lens_ii = rng.choice(np.nonzero(y.reshape(-1))[0] , new_no_lenses, replace=False)
    nonlens_ii = rng.choice(np.nonzero(y.reshape(-1) == 0)[0] , new_no_nonlenses, replace=False)
    train_ii = np.concatenate((lens_ii,nonlens_ii))
    rng.shuffle(train_ii)
    val_ii = np.setdiff1d(np.arange(len(y)), train_ii,assume_unique=False)
    rng.shuffle(val_ii)
    return x[train_ii], x_test, x[val_ii], y[train_ii], y_test, y[val_ii], proportion

    
#%%
# test_size=0.2175
# # test_size=0.3
# # print( train_test_split_with_prevalences(X,y, random_seed=1, test_size=test_size))
# X_train, X_test,  X_val, y_train, y_test, y_val2, prevalence = train_test_validation_split_with_prevalences(
#     X,y,random_seed=2, test_size=test_size)

# print(X_train.shape)
# print(X_test.shape)
# print(X_val.shape)

#%%
# lens_ii = rng.choice(np.nonzero(y.reshape(-1))[0] , 3000,replace=False)
# nonlens_ii = rng.choice(np.nonzero(y.reshape(-1) == 0)[0] , 12000, replace=False)
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

    # X_train, X_test, y_train, y_test, prevalence = train_test_split_with_prevalences(
    #     X,y, random_seed=random_seed, test_size=test_size, proportion=prevalence)

    X_train, X_test, X_val, y_train, y_test, y_val, prevalence = train_test_validation_split_with_prevalences(
        X,y, random_seed=random_seed, test_size=test_size, proportion=prevalence)


    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    X_test = (X_test - np.mean(X_test)) / np.std(X_test) #The example uses kind of a MinMax scaling. TODO: to try that.
    X_val = (X_val - np.mean(X_val)) / np.std(X_val)

    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(y_test.shape))
    return X_train, X_test, X_val, y_train, y_test, y_val, prevalence