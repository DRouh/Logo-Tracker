from sklearn.cluster import KMeans
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import cv2
import imtools
import colorsys

class DominantColoursDetector:
    def __init__(self,imgsPath, colorsNumber = 3, show_clustering_result = False):
        self.Path = imgsPath
        self.Show_clustering_result = show_clustering_result
        self.ColorsNumber = colorsNumber
    
    def findDominantColors(self):                        
        imglist = imtools.get_imlist(self.Path)
        print "Images found %d. Finding dominant colors in set of images" % len(imglist)
        
        clt = KMeans(n_clusters = self.ColorsNumber)
        for i in range(len(imglist)):
            print 'evaluating colors: %d / %d\r' % (i+1, len(imglist)),
            # load the image and convert it from BGR to RGB so that
            # we can dispaly it with matplotlib
            im = cv2.imread(imglist[i]) #cv2.COLOR_BGR2HSV,cv2.COLOR_BGR2RGB
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            
            # reshape the image to be a list of pixels
            im = im.reshape((im.shape[0] * im.shape[1], 3))
            
            # cluster the pixel intensities        
            clt.fit(im)
            
            # build a histogram of clusters and then create a figure
            # representing the number of pixels labeled to each color
            hist = imtools.centroid_histogram(clt)
            bar = imtools.plot_colors(hist, clt.cluster_centers_)
            
            # show our image
            if self.Show_clustering_result == True:
                plt.figure()
                plt.title(os.path.basename(imglist[i]))
                plt.axis("off")
                plt.imshow(bar)
                plt.show()
            #sort by descending    
            sortedColors = np.array(clt.cluster_centers_[np.argsort(-hist)])
        
            #save dominant colors in HSV color space
            hsvColors = np.array([colorsys.rgb_to_hsv(r/255, g/255, b/255) for r, g, b in sortedColors])
            
            #hsvColors[:,0] *= 359 #multiply first column,e.g. hue by 359. Since percnt to absolute
            hsvColors[:,0] *= 179
            hsvColors[:,1] *= 255
            hsvColors[:,2] *= 255            
            np.savetxt(imglist[i][:-4] + '.txt', hsvColors)
        print   
        print "All images proccessed"