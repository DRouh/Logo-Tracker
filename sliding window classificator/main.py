import copy
from skimage.transform import pyramid_gaussian
import numpy as np
import cv2
import time
import imutils


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
    # load the image
    sift = cv2.SIFT()
    image = cv2.imread("cocacola.png", cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    show_keypoints = False
    (winW, winH) = (128, 128)

    for resized in pyramid(image, scale=1.5):
        #calculate keypoints
        cop = resized.copy()
        gray = cv2.cvtColor(cop, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW

            # since we do not have a classifier, we'll just draw the window
            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            if show_keypoints:
                clone = cv2.drawKeypoints(clone, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("Window", clone)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                cv2.destroyAllWindows()
                break
            if ch == ord('s'):
                show_keypoints = not show_keypoints
