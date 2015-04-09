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

class BaseTracker:
    
  def __init__(self, name, labels, imagesBW, imagesCLR, pathToFile, readFromFile = False):
    print 'BaseTracker started. Using', name
    cv2.namedWindow("ColourTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
    capture = pathToFile if readFromFile else 0
    self.capture = cv2.VideoCapture(capture)         
    self.Pool = ThreadPool(processes = cv2.getNumberOfCPUs())    
    self.RefImagesBW = imagesBW
    self.RefImagesCLR = imagesCLR
    self.Labels = labels
    self.Detector, self.Matcher = self.initilializeDetectorAndMatcher(name)
    self.AsiftMatcher = asiftmatcher.AsiftMatcher(self.Matcher) 
    
    
  def run(self):           
    framenum = 1
    
    hr = 160 # constant for ref-logos resizing
    wr = 120 # constant for ref-logos resizing
    h_orig, w_orig, c_orig = (480, 640, 3)      
    logosCount = len(self.Labels)
      
    features = []
    
    for i in range(len(self.Labels)):
        print "Initial features extraction {0}/{1}".format(i + 1, len(self.Labels)),
        refKp, refDescs = self.AsiftMatcher.affine_detect(self.Detector, self.RefImagesBW[i], mask = None, pool = self.Pool)
        features.append((refKp, refDescs))
    
    videoLength = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    while True:
      f, orig_img = self.capture.read()
      if orig_img == None:
          continue
      print "Frame: {0}/{1}".format(framenum, videoLength),

      img_Sift = copy.deepcopy(orig_img) 
      gray_img = cv2.cvtColor(img_Sift, cv2.COLOR_BGR2GRAY)             
      frameKp, frameDescs = self.AsiftMatcher.affine_detect(self.Detector, gray_img, mask=None, pool=self.Pool)
      vis = np.zeros((max(h_orig, hr * logosCount), w_orig + wr, 3), np.uint8)  
      vis[:h_orig, :w_orig] = orig_img   
      found = 0
      
      found, boxes = self.detectLogo(self.Labels[0], features[0], self.RefImagesBW[0], orig_img, gray_img, img_Sift, frameKp, frameDescs)
      if found > 0 and len(boxes) > 0:
          print "found", self.Labels[0]
          cv2.drawContours(vis, [boxes[0]], 0, (255, 255, 0), 2)                       
          #put ref-logo in container
          vis[i * hr:(i + 1) * hr, w_orig:w_orig + wr] = cv2.resize(self.RefImagesCLR[i], (wr, hr))
      else:
          vis[i * hr:(i + 1) * hr, w_orig:w_orig + wr] = np.zeros((hr, wr, 3), np.uint8) 
      
      cv2.imwrite(str(framenum) + ".jpg",vis)
      framenum += 1
      
      if cv2.waitKey(20) == 27:        
        cv2.destroyAllWindows()    
        self.capture.release()
        break
  
  def detectLogo(self, label, feature,ref_img, orig_img, gray_img, img_Sift, frameKp, frameDescs):
      refKp, refDescs = feature      
           
      scores = []
      found = 0
      fKp = frameKp
      fDes = frameDescs
      boxes = []
      cont = True
      
      while(cont):
          cont = False
          inl, matches, matchedKp = self.AsiftMatcher.asift_match(ref_img, gray_img, refKp, refDescs, fKp, fDes)
          
          if inl == None or matches == None:
              break
          
          score = float(inl)/float(matches)
          if matches >= 150 and score > 0.48:
              cont = True
              found += 1
              box = self.MinRectByMatchedKeypoints(matchedKp)
              scores.append(score)
              boxes.append(box)
              fKp, fDes = self.DeleteKeypoints(fKp, fDes, box[1][0], box[1][1], box[3][0], box[3][1])
      print found, len(boxes)                    
      return found, boxes  
  
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