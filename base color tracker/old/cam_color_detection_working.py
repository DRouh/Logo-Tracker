#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2, math
import numpy as np
import colorsys
class ColourTracker:
    
  def __init__(self):
    cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
    self.capture = cv2.VideoCapture(1)
    self.scale_down = 4
#there defined function for getting intersected bounding rects, but it won't work for intersecting boxes    
  def intersect(self,rect1,rect2):
    (x,y),(w,h),a = rect1    
    (x2,y2),(w2,h2),b = rect2
    left = max(x, x2)
    right = min(x + w, x2 + w2)
    top = max(y, y2)
    bottom = max(y + h, y + h2)
    return int(left),int(top),int(right),int(bottom)
  
  def run(self):    
    colors = np.array([175,110])
    while True:
      f, orig_img = self.capture.read()
      orig_img = cv2.flip(orig_img, 1) 
      result_img = orig_img.copy()
      
      img = cv2.GaussianBlur(orig_img, (5,5), 0)
      img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
      img = cv2.resize(img, (len(orig_img[0]) / self.scale_down, len(orig_img) / self.scale_down))
      
      boxArray = np.array([self.getBoundingBox(img, col) for col in colors])
      
      
      for i in range(len(boxArray)):
          if not boxArray[i] == None:              
              r, g, b = (i * 255 for i in colorsys.hsv_to_rgb(colors[i]/float(179),1,1))
              cv2.drawContours(orig_img,[boxArray[i][0]], 0, (b, g, r), 1)
      
      #find correct boxes
      boxIds = np.where(np.array([1.0*(i == None) for i in boxArray]) == 0)[0]
      
      rects = []
      for i in range(len(boxIds)):
        rects.append(boxArray[i][1])
        

      if len(rects) == 2:
          (x1,y1), (w1,h1), a = rects[0]    
          (x2,y2), (w2,h2), b = rects[1] 
          cv2.rectangle(result_img, (int(x1),int(y1)), (int(x1+w1),int(y1+h1)),(0,0,255),2)
          cv2.rectangle(result_img, (int(x2),int(y2)), (int(x2+w2),int(w2+h2)),(0,0,0),2)
          print 'yes'
          print rects
          x,y,w,h = self.intersect(rects[0],rects[1])
          print x,y,w,h          
          cv2.rectangle(result_img, (x,y), (w,h),(255,255,255),2)
      
      cv2.imshow("ColourTrackerWindow", orig_img)
      cv2.imshow("Result", result_img)
      if cv2.waitKey(20) == 27:
        cv2.destroyWindow("ColourTrackerWindow")
        cv2.destroyWindow("Result")
        self.capture.release()
        break
    
  def getBoundingBox(self, bluredimage, hue):
      lower = np.array([max(hue-10,0), 150, 50])
      upper = np.array([min(hue+10,179), 255, 255])
      binary = cv2.inRange(bluredimage, lower, upper)
      
      #dilate binary image to get wider contours
      dilation = np.ones((15, 15), "uint8")
      binary = cv2.dilate(binary, dilation)
      
      #While finding contours, you are also finding the hierarchy of the contours.
      #Hierarchy of the contours is the relation between different contours.
      #So the flag you used in your code, cv2.RETR_TREE provides all the hierarchical relationship.
      #cv2.RETR_LIST provides no hierarchy while cv2.RETR_EXTERNAL gives you only external contours.
      
      contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
      max_area = 0
      largest_contour = None
      
      for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > max_area:
          max_area = area
          largest_contour = contour
          
      if not largest_contour == None:         
        moment = cv2.moments(largest_contour)
        if moment["m00"] > 1000 / self.scale_down:
          rect = cv2.minAreaRect(largest_contour)          
          rect = ((rect[0][0] * self.scale_down, rect[0][1] * self.scale_down), (rect[1][0] * self.scale_down, rect[1][1] * self.scale_down), 0)#rect[2])
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)
          return box,rect
      else:
          return None

  
if __name__ == "__main__":
  colour_tracker = ColourTracker()
  colour_tracker.run()