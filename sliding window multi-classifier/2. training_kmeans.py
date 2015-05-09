from timeit import Timer
import numpy as np
from sklearn.metrics import *
from sklearn.cluster import KMeans
import glob
import pickle
import gc
import os
import sys
# Load the images
import time

if __name__ == '__main__':

    all_instance_filenames = []
    for f in glob.glob('rootsift (descs for kmeans train)/*.rsift'):
        all_instance_filenames.append(f)

    surf_features = []
    counter = 1
    X_train_surf_features = np.empty(shape=[0, 128])
    program_starts = time.time()

    for f in all_instance_filenames:
        print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_filenames))
        x = np.array(pickle.load(open(f, "rb")))
        X_train_surf_features = np.concatenate([X_train_surf_features, x])
        counter += 1

    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))

    # Split the images into training and testing data
    train_len = len(all_instance_filenames)
    indices = np.random.permutation(train_len)
    training_idx = indices[:train_len]

    #X_train_surf_features = np.concatenate([surf_features[i] for i in training_idx])

    # Group the extracted descriptors into 300 clusters using KMeans
    n_clusters = 300
    print 'Clustering', len(X_train_surf_features), 'features'
    program_starts = time.time()
    estimator = KMeans(n_clusters = n_clusters, verbose=1,  n_init=1)
    estimator.fit_transform(X_train_surf_features)

    now = time.time()
    print("It has been {0} seconds since the loop started".format(now - program_starts))
    with open('KMeansColaPepsiAndNoneRootSIFT.pkl', 'wb') as f:
        pickle.dump(estimator, f)


