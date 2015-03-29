import cv2
import numpy as np
import math
cap = cv2.VideoCapture(0)

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV

    h = 332
    s = 30
    v = 57
    h1 = 359
    s1 = 100
    v1 = 80
    lower_blue = np.array([math.floor(179 * h / 359), math.floor(255 * s / 100) , math.floor(255 * v / 100)])
    upper_blue = np.array([math.floor(179 * h1 / 359), math.floor(255 * s1 / 100 ), math.floor(255* v1 / 100)])
    #lower_blue = np.array([110,50,50])
    #upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()