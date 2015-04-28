from os.path import basename
import pickle
from shutil import copyfile, copy
from time import sleep
from rootsift import RootSIFT
import cv2
import numpy as np
import imutils
import glob


def filterKeypoints(kp, des, left_x, left_y, right_x, right_y):
    """Filters given SIFT-keypoints by rect and return them and desc"""
    if kp is None:
        return None, None

    filtered_keypoints = []
    filtered_desc = []
    for i in range(len(kp)):
        kp_x, kp_y = kp[i].pt
        if left_x <= kp_x <= right_x and left_y <= kp_y <= right_y:
            filtered_keypoints.append(kp[i])
            filtered_desc.append(des[i])

    return filtered_keypoints, np.array(filtered_desc)


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':
    # load k-means
    n_clusters = 300
    estimator = pickle.load(open('KMeansCoceAndNoneRootSIFT.pkl', "rb"))

    # load classifier
    clf = pickle.load(open('GB_best.pkl', 'rb'))

    sift = cv2.SIFT()
    rs = RootSIFT()

    # size of sliding-window
    (winW, winH) = (128, 128)
    pathToSaveHard = 'e:/master thesis/Logo-Tracker/sliding window classificator/RootSift/hardnegativemining_base/hard/'
    pathToSaveNotFound = 'e:/master thesis/Logo-Tracker/sliding window classificator/RootSift/hardnegativemining_base/notfound/'
    # Load the images
    all_instance_filenames = []
    for f in glob.glob('hardnegativemining_base/*.jpg'):
        all_instance_filenames.append(f)

    counter = 0
    for f in all_instance_filenames:
        counter += 1
        print "Computing {0}/{1}".format(counter, len(all_instance_filenames))
        im = cv2.imread(f)
        fname = f
        i = 0
        found = 0
        for resized in pyramid(im, scale=1.5):
            # calculate keypoints
            cop = resized.copy()
            gray = cv2.cvtColor(cop, cv2.COLOR_BGR2GRAY)
            kp = sift.detect(gray)
            (kps, des) = rs.compute(gray, kp)
            boxes = []
            for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
                i += 1
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue

                # take only keypoints that belong to the sliding-window
                fkp, fds = filterKeypoints(kp, des, x, y, x + winW, y + winH)

                if fds.any():
                    # calculate b-o-w
                    clusters = estimator.predict(fds)
                    features = np.bincount(clusters)
                    if len(features) < n_clusters:
                        features = np.append(features, np.zeros((1, n_clusters - len(features))))
                    prediction = clf.predict(features)
                else:
                    prediction = [0]

                if prediction == [1]:
                    found += 1
                    crop_img = resized.copy()
                    crop_img = crop_img[x:x + winW, y:y + winH]
                    name = pathToSaveHard + basename(fname)[:-4] + '_' + str(i) + ".jpg"
                    cv2.imwrite(name, crop_img)
                    boxes.append((x, y, x + winW, y + winH))

                # since we do not have a classifier, we'll just draw the window
                # clone = resized.copy()
                # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

                # if len(boxes) > 0:
                # for (_, (x0, y0, x1, y1)) in enumerate(boxes):
                # cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 255), 2)
                #cv2.imshow("Window", clone)
                ch = 0xFF & cv2.waitKey(5)
                if ch == 27:
                    cv2.destroyAllWindows()
                    break
        if found == 0:
            copy(fname, pathToSaveNotFound)