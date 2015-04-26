from sklearn.metrics import *
import numpy as np
import cv2
import pickle

n_clusters = 300
sift = cv2.SIFT() 
estimator = pickle.load(open('KMeansCoceAndNoneSIFT.pkl', "rb"))

img = cv2.imread("cocacola.jpg")
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kp, des = sift.detectAndCompute(gray, None)

surf_features = des
clusters = estimator.predict(surf_features)
features = np.bincount(clusters)        
if len(features) < n_clusters:
    features = np.append(features, np.zeros((1, n_clusters-len(features))))
        
clf = pickle.load(open("GB_best.pkl","rb"))  
predictions = clf.predict(features)
print predictions