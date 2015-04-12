import numpy as np
import os
import sys
import cv2

from tracker.cshtracker import CshTracker


def loadImagesAndLabels(path,loadOnlyHues=True):
    """Loads labels and corresponding reference images"""
    files = [f for f in os.listdir(path) if f.endswith('.png') or f.endswith('.jpg')]
    
    labels = [f[:-4] for f in files]
        
    refer_imgs_bw = [cv2.imread(os.path.join(path,img), 0) for img in os.listdir(path) 
                  if img[:-4] in labels and (img.endswith('.jpg') or img.endswith('.png'))]
    refer_imgs_clr = [cv2.imread(os.path.join(path,img), 1) for img in os.listdir(path) 
                  if img[:-4] in labels and (img.endswith('.jpg') or img.endswith('.png'))]                  
    return labels, refer_imgs_bw, refer_imgs_clr
  
if __name__ == "__main__":
  #sys.stdout = open('file', 'w')
  #path to reference logs
  path = 'e:\\master thesis\\Logo-Tracker\\base tracker with restoration\\ReferensImages\\'

  #color tracker options
  base_tracker_enable = True
  readFromFile = True
  pathToVideo = "e:\\master thesis\\Logo-Tracker\\videos\\st_co.mp4" 
  
#base tracker routine        
  if base_tracker_enable:    
      labels, refer_imgs_bw, refer_imgs_clr = loadImagesAndLabels(path)
      
      tracker = CshTracker("sift", labels, refer_imgs_bw, refer_imgs_clr, pathToVideo, readFromFile)
      tracker.run()