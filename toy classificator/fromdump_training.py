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
sys.stdout = open('log.txt', 'w')

#Load the images
try:
    all_instance_filenames = []
    all_instance_targets = []
    for f in glob.glob('asift_train/*.asift'):
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
    train_len = int(len(all_instance_filenames)*.100)
    indices = np.random.permutation(len(all_instance_filenames))
    training_idx, test_idx = indices[:train_len], indices[train_len:]
    
    gc.collect()
    
    X_train_surf_features = np.concatenate([surf_features[i] for i in training_idx])
    y_train = [all_instance_targets[i]  for i in training_idx]
    
    gc.collect()
    
    #Group the extracted descriptors into 300 clusters. Use MiniBatchKMeans
    #to compute the distances to the centroids for a sample of the instances.
    n_clusters = 300
    print 'Clustering', len(X_train_surf_features), 'features'
    estimator = KMeans(n_clusters=n_clusters, verbose=1)
    estimator.fit_transform(X_train_surf_features)
    with open('KMeansCoceAndNone.pkl','wb') as f:
        pickle.dump(estimator,f)
    #estimator = pickle.load(open('estimator.pkl', "rb"))
    
    X_train = []
    for instance in surf_features[:train_len]:
    	clusters = estimator.predict(instance)
    	features = np.bincount(clusters)
    	if len(features) < n_clusters:
    		features = np.append(features, np.zeros((1, n_clusters-len(features))))
    	X_train.append(features)
    
    #X_test = []
    #count = 0
    #for instance in surf_features[train_len:]:
    #    clusters = estimator.predict(instance)
    #    features = np.bincount(clusters)
    #    if len(features) < n_clusters:  
    #        features = np.append(features, np.zeros((1, n_clusters-len(features))))
    #    X_test.append(features)
    #    count+=1
    
    #Train a logistic regression classifier on the feature vectors and targets,
    #and assess its precision, recall, and accuracy.
    print "Logistic regression"
    clf = LogisticRegression(C=0.001, penalty='l2')
    clf.fit_transform(X_train, y_train)
    #predictions = clf.predict(X_test)
    print classification_report(y_test, predictions)
    print 'Precision: ', precision_score(y_test, predictions)
    print 'Recall: ', recall_score(y_test, predictions)
    print 'Accuracy: ', accuracy_score(y_test, predictions)
    with open('LogisticRegressionCoceAndNone.pkl','wb') as f:
        pickle.dump(clf,f)
    os.system("shutdown /s")
except:
    os.system("shutdown /s")    