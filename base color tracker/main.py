import numpy as np
import os
import cv2

from tracker.colourtracker import ColourTracker

from dominantcolours.dominantcoloursdetector import DominantColoursDetector

def loadColorsAndLabels(path,loadOnlyHues=True,blackAndWhite=True):
    """Load text files with dominant colors, loads labels and corresponding reference images"""
    text_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    
    colors = [np.loadtxt(os.path.join(path, f)) for f in text_files]

    if loadOnlyHues:
      colors = [col[:,0] for col in colors]
      
    labels = [f[:-4] for f in text_files]
    
    bw = 0 if blackAndWhite else 1
    refer_imgs = [cv2.imread(os.path.join(path,img), bw) for img in os.listdir(path) 
                  if img[:-4] in labels and (img.endswith('.jpg') or img.endswith('.png'))]        
    return colors,labels,refer_imgs
  
if __name__ == "__main__":
  path = 'e:/master thesis/Logo-Tracker/base color tracker/Images for color clustering/'
  colorsPath = 'e:/master thesis//Logo-Tracker/base color tracker/Images for color clustering/'
  detect_dominant_colors = False
  color_tracker_enable = True
  
  if detect_dominant_colors:
      colorDetect = DominantColoursDetector(path, show_clustering_result = False)
      colorDetect.findDominantColors()
      
  colors, labels, refer_imgs = loadColorsAndLabels(colorsPath)  
  
  if color_tracker_enable:    
      colour_tracker = ColourTracker("sift", labels, refer_imgs, colors)
      colour_tracker.run()