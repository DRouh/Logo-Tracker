import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report
from multiprocessing.pool import ThreadPool
import glob
import pickle
import cv2
from asiftmatching import asiftmatcher
#sys.stdout = open('log.txt', 'w')

#Load the images
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('t/*.asift'):
	target = 1 if 'cocacola' in f else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)
print 
#Convert the images to grayscale, and extract the SURF descriptors
surf_features = []
counter = 1
detector, matcher = cv2.SIFT(), cv2.BFMatcher(cv2.NORM_L2)
asiftMatcher = asiftmatcher.AsiftMatcher(matcher)
Pool = ThreadPool(processes = 2)#cv2.getNumberOfCPUs())    
for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_targets))     
    surf_features.append(pickle.load(open(f, "rb"))  )   
    counter += 1

#Split the images into training and testing data
train_len = int(len(all_instance_filenames)*.60)

indices = np.random.permutation(len(all_instance_filenames))
training_idx, test_idx = indices[:train_len], indices[train_len:]
print training_idx, test_idx

X_train_surf_features = np.concatenate([surf_features[i] for i in training_idx])
X_test_surf_features = np.concatenate([surf_features[i] for i in test_idx])
y_train = [all_instance_targets[i]  for i in training_idx]
y_test = [all_instance_targets[i]  for i in test_idx]

#Group the extracted descriptors into 300 clusters. Use MiniBatchKMeans
#to compute the distances to the centroids for a sample of the instances.
n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'
estimator = MiniBatchKMeans(n_clusters=n_clusters, verbose=1)
estimator.fit_transform(X_train_surf_features)
#with open('KMeansLgo.pkl','wb') as f:
#    pickle.dump(estimator,f)
#estimator = pickle.load(open('estimator.pkl', "rb"))


X_train = []
for instance in surf_features[:train_len]:
	clusters = estimator.predict(instance)
	features = np.bincount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters-len(features))))
	X_train.append(features)

print "len(surf_features[train_len:])",len(surf_features[train_len:])
X_test = []
count = 0
for instance in surf_features[train_len:]:
    clusters = estimator.predict(instance)
    features = np.bincount(clusters)
    if len(features) < n_clusters:  
        features = np.append(features, np.zeros((1, n_clusters-len(features))))
    X_test.append(features)
    count+=1

#Train a logistic regression classifier on the feature vectors and targets,
#and assess its precision, recall, and accuracy.
print "Logistic regression"
clf = LogisticRegression(C=0.001, penalty='l2')
clf.fit_transform(X_train, y_train)
predictions = clf.predict(X_test)
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)
with open('LogisticRegression.pkl','wb') as f:
    pickle.dump(clf,f)
    
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)