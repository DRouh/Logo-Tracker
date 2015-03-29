import cv2
import numpy as np
import math

# Take each frame

frame = cv2.imread("front1280.jpg")
frame = cv2.medianBlur(frame,5)
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV

h0 = 332
s0 = 30
v0 = 57
h1 = 359
s1 = 100
v1 = 80


h2 = 10
s2 = 10
v2 = 57

h3 = 40
s3 = 40
v3 = 80


lower_first = np.array([math.floor(179 * h0 / 359), math.floor(255 * s0 / 100) , math.floor(255 * v0 / 100)])
upper_first = np.array([math.floor(179 * h1 / 359), math.floor(255 * s1 / 100 ), math.floor(255 * v1 / 100)])

lower_second = np.array([math.floor(179 * h2 / 359), math.floor(255 * s2 / 100) , math.floor(255 * v2 / 100)])
upper_second = np.array([math.floor(179 * h3 / 359), math.floor(255 * s3 / 100 ), math.floor(255 * v3 / 100)])

#lower_blue = np.array([110,50,50])
#upper_blue = np.array([130,255,255])
# Threshold the HSV image to get only blue colors
mask1 = cv2.inRange(hsv, lower_first, upper_first)
mask2 = cv2.inRange(hsv, lower_second, upper_second)

# Bitwise-AND mask and original image
res1 = cv2.bitwise_and(frame,frame, mask = mask1)
res2 = cv2.bitwise_and(frame,frame, mask = mask2)
res = cv2.addWeighted(res1,1,res2,1,0)
#frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
#cv2.imshow('frame',frame)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)

#cv2.imshow('img-windows',res)

cv2.imwrite('masked1.png',res1)
cv2.imwrite('masked2.png',res2)
cv2.imwrite('masked.png',res)