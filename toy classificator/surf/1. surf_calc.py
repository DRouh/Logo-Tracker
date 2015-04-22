import glob
import pickle
import cv2
import mahotas as mh
from mahotas.features import surf
#Load the images
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('train (all pics)/*.jpg'):
	target = 1 if 'cocacola' in f else 0
	all_instance_filenames.append(f)
	all_instance_targets.append(target)
print 
#Convert the images to grayscale, and extract the SIFT descriptors
surf_features = []
counter = 1
for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_targets))
    image = mh.imread(f, as_grey=True)
    des = surf.surf(image)[:,5:]
    fname = f + ".surf"
    with open(fname,'wb') as f:
        pickle.dump(des,f) 
    counter += 1
