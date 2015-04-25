# import the necessary packages
from skimage.transform import pyramid_gaussian
import cv2
import time


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


if __name__ == '__main__':
    # load the image
    image = cv2.imread("cocacola.jpg")
    image = cv2.resize(image, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    (winW, winH) = (128, 128)

    for (i, resized) in enumerate(pyramid_gaussian(image, downscale=1.5)):
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
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            # time.sleep(0.025)