#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import colorsys
from multiprocessing.pool import ThreadPool
import copy

from asiftmatching import asiftmatcher

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

class ColourTracker:
    
  def __init__(self, name, labels, imagesBW, imagesCLR, colors, pathToFile, readFromFile = False):
    print 'ColourTracker started. Using', name
    cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
    capture = pathToFile if readFromFile else 0
    self.capture = cv2.VideoCapture(capture)     
    self.scale_up = 2
    self.Pool = ThreadPool(processes = cv2.getNumberOfCPUs())    
    self.RefImagesBW = imagesBW
    self.RefImagesCLR = imagesCLR
    self.Labels = labels
    self.Colors = colors        
    self.Detector, self.Matcher = self.initilializeDetectorAndMatcher(name)
    self.AsiftMatcher = asiftmatcher.AsiftMatcher(self.Matcher) 
    
    
  def run(self):           
    framenum = 0
    
    hr = 160 # constant for ref-logos resizing
    wr = 120 # constant for ref-logos resizing
    h_orig, w_orig, c_orig = (480, 640, 3)      
    logNum = len(self.Colors)
      
    
    length = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    while True:
      f, orig_img = self.capture.read()
      if orig_img == None:
          continue
      print "Frame: {0}/{1}".format(framenum + 1, length),
 
      #for i in range(3):
      #    orig_img[:, :, i] = cv2.equalizeHist(orig_img[:, :, i])       
      img_Sift = copy.deepcopy(orig_img)    
      
      #calculate sift and draw it on result_img
      gray_img = cv2.cvtColor(img_Sift, cv2.COLOR_BGR2GRAY)             
      frameKp, frameDescs = self.AsiftMatcher.affine_detect(self.Detector, gray_img, mask=None, pool=self.Pool)
      
      #put frame in container
      vis = np.zeros((max(h_orig, hr * logNum), w_orig + wr, 3), np.uint8)  
      vis[:h_orig, :w_orig] = orig_img   
      found = 0
      for i in range(len(self.Labels)):
          found, box = self.detectLogo(self.Labels[i], self.Colors[i], self.RefImagesBW[i], orig_img, gray_img, img_Sift, frameKp, frameDescs)
          #put logo ref-image along with frame if it's found in it
          if found > 0 and len(box) > 0:
              print "found", self.Labels[i]
              cv2.drawContours(vis,[box], 0, (255, 255, 0), 2)                       
              #put ref-logo in container
              vis[i * hr:(i + 1) * hr, w_orig:w_orig + wr] = cv2.resize(self.RefImagesCLR[i], (wr, hr))
          else:
              vis[i * hr:(i + 1) * hr, w_orig:w_orig + wr] = np.zeros((hr, wr, 3), np.uint8) 
      
      #cv2.imshow("sift", gray_img)
      #cv2.imshow("ColourTrackerWindow", orig_img) 

      cv2.imwrite(str(framenum) + ".jpg",vis)
      framenum += 1
      
      if cv2.waitKey(20) == 27:        
        cv2.destroyAllWindows()    
        self.capture.release()
        break
  
  def detectLogo(self, label, colors, ref_img, orig_img, gray_img, img_Sift, frameKp, frameDescs):
      refKp, refDescs = self.AsiftMatcher.affine_detect(self.Detector, ref_img, mask = None, pool = self.Pool)
      img = cv2.GaussianBlur(orig_img, (5, 5), 0)
      img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)            
      boxArray = np.array([self.getBoundingBox(img, int(col), resize=True) for col in colors])
           
      scores = []
      found = 0
      fKp = frameKp
      fDes = frameDescs
      boxes = []

      for i in range(len(boxArray)):
          if boxArray[i] != None:              
              r, g, b = (i * 255 for i in colorsys.hsv_to_rgb(colors[i] / float(179), 1, 1))                               
              filtered_keypoints, filtered_descs = self.filterKeypoints(
                  fKp, fDes, boxArray[i][1][0], boxArray[i][1][1], boxArray[i][3][0], boxArray[i][3][1])

              if filtered_keypoints != None and filtered_descs != None:                
                #gray_img = cv2.drawKeypoints(gray_img, filtered_keypoints, filtered_descs)  
                #cv2.drawContours(gray_img,[boxArray[i]], 0, (b, g, r), 1) 
                                                              
                inl, matches, matchedKp = self.AsiftMatcher.asift_match(ref_img, gray_img, refKp, refDescs, filtered_keypoints, filtered_descs)
                
                if inl == None or matches == None:
                  continue
            
                score = float(inl)/float(matches)
            
                if matches >= 150 and score > 0.48:
                  print matches
                  found += 1
                  box = self.MinRectByMatchedKeypoints(matchedKp) 
                  scores.append(score)
                  boxes.append(box)                                          
                  fKp, fDes = self.DeleteKeypoints(
                      fKp, fDes, boxArray[i][1][0], boxArray[i][1][1], boxArray[i][3][0], boxArray[i][3][1])
              #cv2.drawContours(orig_img,[boxArray[i]], 0, (b, g, r), 1)                            
      #cv2.imshow("sift", gray_img)            
      
      if len(scores) > 0:
          maxIdx = np.argmax(np.array(scores))
          print scores[maxIdx]
          return found, boxes[maxIdx]
      else:
          return 0, 0
  
  def MinRectByMatchedKeypoints(self, kp):
      """Constructs min rect by matched keypoints. 
      kp - array of x,y coordinates, not opencv keypoints"""
      points = kp #np.array([p.pt for p in kp])
      minX, minY =  np.amin(points, axis=0)
      maxX, maxY =  np.amax(points, axis=0)
      box = []
      box.append([minX, maxY])
      box.append([minX, minY])      
      box.append([maxX, minY])      
      box.append([maxX, maxY])      
      return np.int0(box)
            
  def DeleteKeypoints(self, kp, des, left_x, left_y, right_x, right_y):
      """Deletes keypoints from set so we can check only new keypoints in that frame"""
      if kp == None:
          return None, None
          
      filtered_keypoints = []
      filtered_desc = []
      for i in range(len(kp)):
          kp_x,kp_y = kp[i].pt
          if(kp_x < left_x and kp_x > right_x and kp_y < left_y and kp_y > right_y):
            filtered_keypoints.append(kp[i])
            filtered_desc.append(des[i])
            
      return filtered_keypoints, np.array(filtered_desc)   
  
  def filterKeypoints(self, kp, des, left_x, left_y, right_x, right_y):
      """Filters given SIFT-keypoints by rect and return them and desc"""
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
          
  def getBoundingBox(self, bluredimage, hue, resize = False):
      lower = np.array([max(hue - 5, 0), 20, 20])
      upper = np.array([min(hue + 10, 180), 255, 255])
      
      if resize:
          bluredimage = cv2.resize(bluredimage, (len(bluredimage[0]) * self.scale_up, len(bluredimage) * self.scale_up))
          
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
          
      if largest_contour != None:         
        moment = cv2.moments(largest_contour)
        factor = self.scale_up if resize else 1
        if moment["m00"] > 750 * factor:
          rect = cv2.minAreaRect(largest_contour)                    
          rect = ((rect[0][0] / factor, rect[0][1] / factor), (rect[1][0] / factor, rect[1][1] / factor), 0)#rect[2])
          box = cv2.cv.BoxPoints(rect)
          box = np.int0(box)
          return box
      else:
          return None
          
  def initilializeDetectorAndMatcher(self, name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.SIFT()
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.SURF(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher