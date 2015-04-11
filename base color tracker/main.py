import numpy as np
import os
import cv2

from tracker.colourtracker import ColourTracker
from dominantcolours.dominantcoloursdetector import DominantColoursDetector

def loadColorsAndLabels(path,loadOnlyHues=True):
    """Load text files with dominant colors, loads labels and corresponding reference images"""
    text_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    
    colors = [np.loadtxt(os.path.join(path, f)) for f in text_files]

    if loadOnlyHues:
      colors = [col[:,0] for col in colors]
      
    labels = [f[:-4] for f in text_files]
        
    refer_imgs_bw = [cv2.imread(os.path.join(path,img), 0) for img in os.listdir(path) 
                  if img[:-4] in labels and (img.endswith('.jpg') or img.endswith('.png'))]
    refer_imgs_clr = [cv2.imread(os.path.join(path,img), 1) for img in os.listdir(path) 
                  if img[:-4] in labels and (img.endswith('.jpg') or img.endswith('.png'))]                  
    return colors,labels, refer_imgs_bw, refer_imgs_clr
  
if __name__ == "__main__":
  #dominant colors options
  path = 'e:/master thesis/Logo-Tracker/base color tracker/Images for color clustering/'
  colorsPath = 'e:/master thesis//Logo-Tracker/base color tracker/Images for color clustering/'
  detect_dominant_colors = False
  
  #color tracker options
  color_tracker_enable = True
  readFromFile = True
  pathToVideo = "e:/master thesis/Logo-Tracker/videos/bigcoce1.mp4"
  
#detect dominant colors
  if detect_dominant_colors:
      colorDetect = DominantColoursDetector(path, show_clustering_result = False)
      colorDetect.findDominantColors()

#color tracker routine        
  if color_tracker_enable:    
      colors, labels, refer_imgs_bw, refer_imgs_clr, = loadColorsAndLabels(colorsPath)
      print labels
      colour_tracker = ColourTracker("sift", labels, refer_imgs_bw, refer_imgs_clr, colors, pathToVideo, readFromFile)
      colour_tracker.run()