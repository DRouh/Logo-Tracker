import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
import glob
import pickle
import os

import sys
sys.stdout = open('log', 'w')
print 'Started'

#Load the images
train_instance_filenames = []
train_instance_targets = []
test_instance_filenames = []
test_instance_targets = []
for f in glob.glob('train/*.jpg'):
	target = 1 if 'cocacola' in f else 0
	train_instance_filenames.append(f)
	train_instance_targets.append(target)
print 

for f in glob.glob('test/*.jpg'):
	target = 1 if 'cocacola' in f else 0
	test_instance_filenames.append(f)
	test_instance_targets.append(target)
print 

#Convert the images to grayscale, and extract the SURF descriptors
train_surf_features = []
test_surf_features = []
counter = 1
for f in train_instance_filenames:
    print 'Reading train image:{0}. {1}/{2}'.format(f, counter, len(train_instance_targets))
    image = mh.imread(f, as_grey=True)
    train_surf_features.append(surf.surf(image)[:,5:])
    counter += 1

counter = 1
for f in test_instance_filenames:
    print 'Reading test image:{0}. {1}/{2}'.format(f, counter, len(test_instance_targets))
    image = mh.imread(f, as_grey=True)
    test_surf_features.append(surf.surf(image)[:,5:])
    counter += 1
    
X_train_surf_features = np.concatenate([train_surf_features[i] for i in range(0, len(training_idx))])
X_test_surf_features = np.concatenate([test_surf_features[i] for i in range(0, len(training_idx))])
y_train = train_instance_targets
y_test = test_instance_targets

#Group the extracted descriptors into 300 clusters. Use MiniBatchKMeans
#to compute the distances to the centroids for a sample of the instances.
n_clusters = 300
print 'Clustering', len(X_train_surf_features), 'features'

#estimator = pickle.load(open('estimator.pkl', "rb"))

#estimator = KMeans(n_clusters=n_clusters, verbose=1, n_init = 1)
#estimator.fit_transform(X_train_surf_features)
#with open('KMeans.pkl','wb') as f:
#    pickle.dump(estimator,f)

estimator = pickle.load(open('KMeans.pkl', "rb"))

X_train = []
for instance in X_train_surf_features:
	clusters = estimator.predict(instance)
	features = np.bincount(clusters)
	if len(features) < n_clusters:
		features = np.append(features, np.zeros((1, n_clusters-len(features))))
	X_train.append(features)

X_test = []
count = 0
for instance in X_test_surf_features:
    print count
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

with open('LogisticRegression.pkl','wb') as f:
    pickle.dump(clf,f)
    
print classification_report(y_test, predictions)
print 'Precision: ', precision_score(y_test, predictions)
print 'Recall: ', recall_score(y_test, predictions)
print 'Accuracy: ', accuracy_score(y_test, predictions)

print "Started random forest"

pipeline = Pipeline([
    ('clf', RandomForestClassifier(criterion='entropy',verbose=1))
  ])
  
parameters = {
    'clf__n_estimators': (10000,),
    'clf__max_depth': (25,),
    'clf__min_samples_split': (1,),
    'clf__min_samples_leaf': (1,2,3,)
  }
  
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1')
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
#with open('KMeans.pkl','wb') as f:
#    pickle.dump(estimator,f)
    
with open('RF.pkl','wb') as f:
    pickle.dump(grid_search.best_estimator_,f)

os.system("shutdown /s")