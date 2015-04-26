import numpy as np
from sklearn.metrics import *
from sklearn.cluster import KMeans
import glob
import pickle
import gc
import os
import sys
# sys.stdout = open('log.txt', 'w')

#Load the images

all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('rootsift (descs for kmeans train)/*.rsift'):
    target = 1 if 'cocacola' in f else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)
print

surf_features = []
counter = 1
for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_targets))
    surf_features.append(pickle.load(open(f, "rb")))
    counter += 1

#Split the images into training and testing data
train_len = int(len(all_instance_filenames) * .100)
indices = np.random.permutation(len(all_instance_filenames))
training_idx, test_idx = indices[:train_len], indices[train_len:]

gc.collect()

X_train_surf_features = np.concatenate([surf_features[i] for i in training_idx])
y_train = [all_instance_targets[i] for i in training_idx]

gc.collect()

#Group the extracted descriptors into 300 clusters. Use MiniBatchKMeans
#to compute the distances to the centroids for a sample of the instances.
n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'
estimator = KMeans(n_clusters=n_clusters, verbose=1)
estimator.fit_transform(X_train_surf_features)
with open('KMeansCoceAndNoneRootSIFT.pkl', 'wb') as f:
    pickle.dump(estimator, f)
