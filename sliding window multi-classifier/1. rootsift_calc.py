import glob
import pickle
import cv2
from rootsift import RootSIFT

# Load the images
all_instance_filenames = []
all_instance_targets = []
for f in glob.glob('images all/*.jpg'):
    target = 1 if 'cocacola' in f else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)
print
#Convert the images to grayscale, and extract the SIFT descriptors
surf_features = []
counter = 1
sift = cv2.SIFT()
rs = RootSIFT()
for f in all_instance_filenames:
    print 'Reading image:{0}. {1}/{2}'.format(f, counter, len(all_instance_targets))
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray)
    (kps, des) = rs.compute(gray, kp)
    fname = f + ".rsift"
    with open(fname, 'wb') as f:
        pickle.dump(des, f)
    counter += 1
