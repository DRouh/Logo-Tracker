#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import colorsys
import asift
class ColourTracker:
    
  def __init__(self):
    cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
    self.capture = cv2.VideoCapture(0)    
    self.scale_down = 4
  
  def run(self):    
    #colors = np.array([23,11,111])
    sift = cv2.SIFT() 
    colors = np.array([170])
    img1 = cv2.imread("e:\\master thesis\\Logo-Tracker\\base color tracker\\coca-cola.jpg", 0)
    res = cv2.resize(img1, (640, 480), interpolation = cv2.INTER_CUBIC)
    kp1, des1 = sift.detectAndCompute(res, None)   
    
    out = cv2.VideoWriter('out.avi', 1, 12.0, (640,480))
    while True:
      f, orig_img = self.capture.read()
      #orig_img = cv2.flip(orig_img, 1)            
      img_Sift = np.copy(orig_img)
      
      #calculate sift and draw it on result_img
      sift_img = cv2.cvtColor(img_Sift ,cv2.COLOR_BGR2GRAY)       
      kp, des = sift.detectAndCompute(sift_img, None)          
      
      img = cv2.GaussianBlur(orig_img, (5,5), 0)
      img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
      img = cv2.resize(img, (len(orig_img[0]) / self.scale_down, len(orig_img) / self.scale_down))
      
      boxArray = np.array([self.getBoundingBox(img, col) for col in colors])      

      for i in range(len(boxArray)):
          if not boxArray[i] == None:              
              r, g, b = (i * 255 for i in colorsys.hsv_to_rgb(colors[i] / float(179), 1, 1))   
              filtered_keypoints, filtered_descs = self.filterKeypoints(kp, des, boxArray[i][1][0],
                                                                        boxArray[i][1][1], 
                                                                        boxArray[i][3][0], 
                                                                        boxArray[i][3][1])
              if filtered_keypoints != None:
                img_Sift = cv2.drawKeypoints(sift_img, filtered_keypoints, filtered_descs)
                cv2.drawContours(img_Sift,[boxArray[i]], 0, (b, g, r), 1)       
                
                #do asift matching here
                asift.my_asift_detection(img1, sift_img, kp1, des1, filtered_keypoints, filtered_descs)                
              cv2.drawContours(orig_img,[boxArray[i]], 0, (b, g, r), 1)              
              
      out.write(img_Sift)
      cv2.imshow("ColourTrackerWindow", orig_img)
      cv2.imshow("SIFT", img_Sift)
      
      if cv2.waitKey(20) == 27:
        
        cv2.destroyWindow("ColourTrackerWindow")
        cv2.destroyWindow("SIFT")
        out.release()
        self.capture.release()
        break
    
  def filterKeypoints(self,kp, des, left_x, left_y, right_x, right_y):
      """Filters given SIFT-keypoints and return them and desc"""
      if kp == None:
          return None,None
          
      filtered_keypoints = []
      filtered_desc = []
      for i in range(len(kp)):
          kp_x,kp_y = kp[i].pt
          if(kp_x >= left_x and kp_x <= right_x and kp_y >= left_y and kp_y <= right_y):
            filtered_keypoints.append(kp[i])
            filtered_desc.append(des[i])
            
      return filtered_keypoints, np.array(filtered_desc)
  
  def getBoundingBox(self, bluredimage, hue):
      lower = np.array([max(hue - 10, 0), 150, 50])
      upper = np.array([min(hue + 10, 179), 255, 255])
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
          return box
      else:
          return None

  
if __name__ == "__main__":
  colour_tracker = ColourTracker()
  colour_tracker.run()