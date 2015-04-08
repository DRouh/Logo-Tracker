#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2, math
import numpy as np
class CT:
  def __init__(self):
    cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
    #self.capture = cv2.VideoCapture(0)
    self.scale_down = 2
  def run(self):
    #while True:
      #f, orig_img = self.capture.read()
      #orig_img = cv2.flip(orig_img, 1)
      orig_img = cv2.imread("d:/3.jpg")
      img = cv2.GaussianBlur(orig_img, (5,5), 0)
      img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
      img = cv2.resize(img, (len(orig_img[0]) * self.scale_down, len(orig_img) * self.scale_down))
      red_lower = np.array([59, 20, 20])
      red_upper = np.array([69, 255, 255])
      red_binary = cv2.inRange(img, red_lower, red_upper)
      dilation = np.ones((15, 15), "uint8")
      red_binary = cv2.dilate(red_binary, dilation)
      contours, hierarchy = cv2.findContours(red_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      max_area = 0
      largest_contour = None
      for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
          max_area = area
          largest_contour = contour
      if not largest_contour == None:
        moment = cv2.moments(largest_contour)
        if moment["m00"] > 500 * self.scale_down:
          rect = cv2.minAreaRect(largest_contour)
          rect = ((rect[0][0] / self.scale_down, rect[0][1] / self.scale_down), (rect[1][0] / self.scale_down, rect[1][1] / self.scale_down), 0)
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)
          cv2.drawContours(orig_img,[box], 0, (0, 0, 255), 2)
          cv2.imshow("ColourTrackerWindow", orig_img)
       #   cv2.imshow("123",red_binary)
          if cv2.waitKey(20) == 27:
            cv2.destroyAllWindows()
            #self.capture.release()

if __name__ == "__main__":
  colour_tracker = CT()
  colour_tracker.run()