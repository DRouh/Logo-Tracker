import pickle
from time import sleep

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


if __name__ == '__main__':

    # load k-means
    n_clusters = 300
    estimator = pickle.load(open('KMeansCoceAndNoneSIFT.pkl', "rb"))

    # load classifier
    clf = pickle.load(open('GB_best.pkl', 'rb'))

    sift = cv2.SIFT()

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
        kp, des = sift.detectAndCompute(gray, None)
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

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            #print prediction

            if len(boxes) > 0:
                for (_, (x0, y0, x1, y1)) in enumerate(boxes):
                    cv2.rectangle(clone, (x0, y0), (x1, y1), (255, 0, 255), 2)

            if show_keypoints:
                clone = cv2.drawKeypoints(clone, fkp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Window", clone)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                cv2.destroyAllWindows()
                break
            if ch == ord('s'):
                show_keypoints = not show_keypoints
            sleep(0.1)