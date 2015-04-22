import numpy as np
from sklearn.cluster import KMeans
import glob
import pickle

#Load the images
all_instance_filenames = []
for f in glob.glob('asift (all descriptors)/*.asift'):
	target = 1 if 'cocacola' in f else 0
	all_instance_filenames.append(f)
print 

surf_features = []
counter = 1
n_clusters = 300    
estimator = pickle.load(open('KMeansCoceAndNone.pkl', "rb"))
#calc hist for all descsriptors
for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_filenames))     
    surf_features = pickle.load(open(f, "rb"))   
    counter += 1
    train_len = 1
    clusters = estimator.predict(surf_features)
    features = np.bincount(clusters)        
    if len(features) < n_clusters:
        features = np.append(features, np.zeros((1, n_clusters-len(features))))
    fname = f + '.hist'     
    print features
    with open(fname,'wb') as fi:
        pickle.dump(features, fi)
  