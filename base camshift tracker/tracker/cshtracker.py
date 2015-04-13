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
    self.InitialWindows = (0, 0, 640, 480)#change if needed
    
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
        
        self.TrackWindows.append(self.InitialWindows)        
        hsv_roi = hsvref[0:yr, 0:xr]
        mask_roi = maskerf[0:yr, 0:xr]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )        
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        hist = hist.reshape(-1)
        self.Histograms.append(hist)

    
    if cv2.waitKey(20) == 27:        
        cv2.destroyAllWindows()    
        self.capture.release()
#        break
             
  def show_hist(self, hist):
        bin_count = hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
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