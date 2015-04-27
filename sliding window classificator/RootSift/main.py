import pickle
from time import sleep
from rootsift import RootSIFT
import cv2
import numpy as np

import imutils


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


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


if __name__ == '__main__':

    # load k-means
    n_clusters = 300
    estimator = pickle.load(open('KMeansCoceAndNoneRootSIFT.pkl', "rb"))

    # load classifier
    clf = pickle.load(open('GB_best.pkl', 'rb'))

    sift = cv2.SIFT()
    rs = RootSIFT()
    # load the image
    image = cv2.imread("cocacola.png", cv2.COLOR_BGR2GRAY)

    # resize image (ex. is too big)
    image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

    show_keypoints = False

    # size of sliding-window
    (winW, winH) = (128, 128)
    boxes = []
    for resized in pyramid(image, scale=1.5):
        # calculate keypoints
        cop = resized.copy()
        gray = cv2.cvtColor(cop, cv2.COLOR_BGR2GRAY)
        kp = sift.detect(gray)
        (kps, des) = rs.compute(gray, kp)
        boxes = []
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
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
                # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
                # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
                # WINDOW
            else:
                prediction = [0]

            if prediction == [1]:
                boxes.append((x, y, x + winW, y + winH))
            print prediction
            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            # print prediction

            if len(boxes) > 0:
                # perform non-maximum suppression on the bounding boxes
                pick = non_max_suppression_fast(np.array(boxes), 0.1)

                #for (_, (x0, y0, x1, y1)) in enumerate(boxes):
                #    cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 255), 2)

                for (_, (x0, y0, x1, y1)) in enumerate(pick):
                    cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 0), 2)

            if show_keypoints:
                clone = cv2.drawKeypoints(clone, fkp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Window", clone)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                cv2.destroyAllWindows()
                break
            if ch == ord('s'):
                show_keypoints = not show_keypoints
            sleep(0.025)