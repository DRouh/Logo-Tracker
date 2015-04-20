from multiprocessing.pool import ThreadPool
import glob
import pickle
import cv2
from asiftmatching import asiftmatcher

#Load the images
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('t/*.jpg'):
	target = 1 if 'cocacola' in f else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)
print 
#Convert the images to grayscale, and extract the SURF descriptors
surf_features = []
counter = 1
detector, matcher = cv2.SIFT(), cv2.BFMatcher(cv2.NORM_L2)
asiftMatcher = asiftmatcher.AsiftMatcher(matcher)
Pool = ThreadPool(processes = 8) 
for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_targets))
    img = cv2.imread(f)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kp, des = asiftMatcher.affine_detect(detector, gray, mask=None, pool=Pool)
    fname = f + ".asift"
    with open(fname,'wb') as f:
        pickle.dump(des,f) 
    counter += 1
