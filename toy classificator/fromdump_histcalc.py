import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
import glob
import pickle
import gc
import os
import sys
#sys.stdout = open('log.txt', 'w')

#Load the images

all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('asift_train/*.asift'):
	target = 1 if 'cocacola' in f else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)
print 

surf_features = []
counter = 1
n_clusters = 300    
estimator = pickle.load(open('KMeansCoceAndNone.pkl', "rb"))

for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_targets))     
    surf_features = pickle.load(open(f, "rb"))   
    counter += 1
    train_len = 1
#    temp = pickle.load(open(f + '.hist.', "rb"))
#    print temp
    for instance in surf_features[:train_len]:
        clusters = estimator.predict(instance)
        features = np.bincount(clusters)        
        if len(features) < n_clusters:
            features = np.append(features, np.zeros((1, n_clusters-len(features))))
            fname = f + '.hist'     
        print features
        with open(fname,'wb') as fi:
            pickle.dump(features, fi)
  