import sys
import numpy as np
import cv2
import sys

class App(object):
    def __init__(self, video_src):
        self.cam = cv2.VideoCapture(0)#"e:/master thesis/Logo-Tracker/base camshift tracker/st_co.mp4") 
        ret, self.frame = self.cam.read()
        cv2.namedWindow('camshift')
        cv2.setMouseCallback('camshift', self.onmouse)

        self.selection = None
        self.drag_start = None
        self.tracking_state = 0
        self.show_backproj = False

    def onmouse(self, event, x, y, flags, param):
        x, y = np.int16([x, y]) # BUG
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                h, w = self.frame.shape[:2]
                xo, yo = self.drag_start
                x0, y0 = np.maximum(0, np.minimum([xo, yo], [x, y]))
                x1, y1 = np.minimum([w, h], np.maximum([xo, yo], [x, y]))
                self.selection = None
                if x1-x0 > 0 and y1-y0 > 0:
                    self.selection = (x0, y0, x1, y1)
            else:
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1

    def show_hist(self):
        bin_count = self.hist.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count*bin_w, 3), np.uint8)
        for i in xrange(bin_count):
            h = int(self.hist[i])
            cv2.rectangle(img, (i*bin_w+2, 255), ((i+1)*bin_w-2, 255-h), (int(180.0*i/bin_count), 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imshow('hist', img)

    def run(self):
        ref = cv2.imread("e:/master thesis/Logo-Tracker/base camshift tracker/ReferensImages/coca-cola.jpg")
        hsvref = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
        maskerf  = cv2.inRange(hsvref, np.array((0., 30., 30.)), np.array((180., 255., 255.)))
        xr,yr,dr = hsvref.shape
        x0, y0, x1, y1 = (0,0,640,480)#self.selection
        self.track_window = (x0, y0, x1-x0, y1-y0)
        hsv_roi = hsvref[0:yr, 0:xr]
        mask_roi = maskerf[0:yr, 0:xr]
        hist = cv2.calcHist( [hsv_roi], [0], mask_roi, [16], [0, 180] )
        #hist = cv2.calcHist([hsv_roi],[0, 1], None, [256, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX);
        self.hist = hist.reshape(-1)
        
        while True:
            ret, self.frame = self.cam.read()
            if not ret:
                continue
            vis = self.frame.copy()
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
          
            if self.tracking_state == 0:
                self.selection = None
                hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
                prob = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)                
                prob &= mask
                #dilation = np.ones((5, 5), "uint8")
                #prob = cv2.dilate(prob, dilation)
                #prob = cv2.erode(prob, dilation)
                if not prob.any():
                    continue
                else:                    
                    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                    self.track_window = (0,0,640,480) if self.track_window == (0,0,0,0) else self.track_window 
                    track_box, self.track_window = cv2.CamShift(prob, self.track_window, term_crit)
                    print 
                #prob = cv2.calcBackProject([hsv],[0,1],hist,[0,180,0,256],1)
                

                if self.show_backproj:
                    vis[:] = prob[...,np.newaxis]
                #try: 
                pts = cv2.cv.BoxPoints(track_box)
                pts = np.int0(pts)               
                #vis = cv2.polylines(vis,[pts],True, 255,2)
                cv2.drawContours(vis,[pts], 0, (255, 255, 0), 2)  
                #cv2.imshow('img2',img2)
                    #cv2.ellipse(vis, track_box, (0, 0, 255), 2)
                #except: print track_box

            cv2.imshow('camshift', vis)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break
            if ch == ord('b'):
                self.show_backproj = not self.show_backproj
            if ch == ord('h'):    
                self.show_hist = not self.show_hist
        cv2.destroyAllWindows()        


if __name__ == '__main__':    
    try: video_src = sys.argv[1]
    except: video_src = 0
    print __doc__
    App(video_src).run()