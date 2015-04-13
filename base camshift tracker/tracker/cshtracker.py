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

class CshTracker:
    
  def __init__(self, name, labels, imagesBW, imagesCLR, pathToFile, readFromFile = False):
    print 'CshTracker started. Using', name
    cv2.namedWindow("CshTrackerWindow", cv2.CV_WINDOW_AUTOSIZE)
    capture = pathToFile if readFromFile else 0
    
    self.capture = cv2.VideoCapture(capture)         
    self.Pool = ThreadPool(processes = cv2.getNumberOfCPUs())    
    self.RefImagesBW = imagesBW
    self.RefImagesCLR = imagesCLR
    self.Labels = labels
    self.Detector, self.Matcher = self.initilializeDetectorAndMatcher(name)
    self.AsiftMatcher = asiftmatcher.AsiftMatcher(self.Matcher)
    
    self.TrackWindows = []
    self.Histograms = []
    self.InitialWindow = (0, 0, 640, 480)#change if needed
    self.CamShiftTermCriteria = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    
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
        #calculate histograms        
        hsvref = cv2.cvtColor(self.RefImagesCLR[i], cv2.COLOR_BGR2HSV)
        maskerf  = cv2.inRange(hsvref, np.array((0., 30., 30.)), np.array((180., 255., 255.)))
        xr, yr, dr = hsvref.shape
        
        self.TrackWindows.append(self.InitialWindow)        
        hsv_roi = hsvref[0:yr, 0:xr]
        mask_roi = maskerf[0:yr, 0:xr]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )        
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist = hist.reshape(-1)
        self.Histograms.append(hist)
        
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
      
      for j in range(len(self.Labels)):
          found, box = self.detectLogo(j, self.Labels[j], features[j], self.RefImagesBW[j], orig_img, gray_img, img_Sift, frameKp, frameDescs)
          if found > 0:
              print "found", self.Labels[0]
              for i in range(len(box)):
                  cv2.drawContours(vis, [box], 0, (255, 255, 0), 2)                                                   
              vis[j * hr:(j + 1) * hr, w_orig:w_orig + wr] = cv2.resize(self.RefImagesCLR[j], (wr, hr))
              cv2.putText(vis,str(found),(w_orig,(j + 1) * hr), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255))
          else:
              vis[j * hr:(j + 1) * hr, w_orig:w_orig + wr] = np.zeros((hr, wr, 3), np.uint8) 
      
      cv2.imwrite(str(framenum) + ".jpg",vis)
      framenum += 1
    
      if cv2.waitKey(20) == 27:        
          cv2.destroyAllWindows()    
          self.capture.release()
          break

  def detectLogo(self, num, label, feature, ref_img, orig_img, gray_img, img_Sift, frameKp, frameDescs):
      refKp, refDescs = feature                 
      found = 0
      fKp = frameKp
      fDes = frameDescs                   
      
      hsv = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
      mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
      hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
      
      prob = cv2.calcBackProject([hsv], [0], self.Histograms[num], [0, 180], 1)                
      prob &= mask
      track_window = (0,0,640,480) if self.TrackWindows[num] == (0,0,0,0) else self.TrackWindows[num]
      track_box, self.TrackWindows[num] = cv2.CamShift(prob, track_window, self.CamShiftTermCriteria)
      
      pts = cv2.cv.BoxPoints(track_box)
      pts = np.int0(pts)        
      maxX, maxY = pts.max(axis=0)
      minX, minY = pts.min(axis=0)
                
      filteredKps, filteredDescs = self.filterKeypoints(fKp, fDes, minX, minY, maxX, maxY)
      if filteredKps == None:
          self.TrackWindows[num] = self.InitialWindow
          return 0, 0
          
      inl, matches, matchedKp = self.AsiftMatcher.asift_match(ref_img, gray_img, refKp, refDescs, fKp, fDes)    
      if inl == None or matches == None:
          self.TrackWindows[num] = self.InitialWindow
          return 0, 0
          
      print "Inliers: {0}. Mathces: {1}.".format(inl, matches)
      score = float(inl)/float(matches)
      box = None
      if matches >= 100 and (matches >= 200 or score > 0.48 or inl > 100):             
          #cont = True
          found += 1
          box = self.MinRectByMatchedKeypoints(matchedKp)
      else:
          self.TrackWindows[num] = self.InitialWindow
             
      print "Found: {0}. Score: {1}".format(found, score)      
      return found, box 

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
           
  def show_hist(self, hist):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h), (int(180.0 * i / bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)        
        cv2.imshow("hui", img)
        
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