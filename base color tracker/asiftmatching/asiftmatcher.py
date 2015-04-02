# import the necessary packages
import numpy as np
import cv2
import itertools as it

# local modules
from common import Timer

class AsiftMatcher:
    def __init__(self,matcher):		
        print "Initializing asift matcher"
        self.Mactcher = matcher

    def asift_match(self, im1, im2, kp1, desc1, kp2, desc2):
        img1 = im1
        img2 = im2
        norm = cv2.NORM_L2        
        
        def match_and_draw(win):
            noMatches = False
            kp_pairs = None
            H = None
            if(len(desc2) > 1):
                with Timer('matching'):                
                    raw_matches = self.Mactcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
                    
                p1, p2, kp_pairs = self.filter_matches(kp1, kp2, raw_matches)
                if len(p1) >= 4 and len(p2) >= 4:
                    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
                    print '%d / %d  inliers/matched' % (np.sum(status), len(status))
                    # do not draw outliers (there will be a lot of them)
                    kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]                    
                else:
                    H, status = None, None
                    noMatches = True
                    print '%d matches found, not enough for homography estimation' % len(p1)
            else:
                noMatches = True
    
            win, vis = self.explore_match(win, img1, img2, kp_pairs, noMatches, None, H)
            return win, vis
            
        win, vis = match_and_draw('affine find_obj')
        return win, vis
        
    def affine_skew(self,tilt, phi, img, mask=None):
        '''
        affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai
        Ai - is an affine transform matrix from skew_img to img
        '''
        h, w = img.shape[:2]
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
            mask[:] = 255
        A = np.float32([[1, 0, 0], [0, 1, 0]])
        if phi != 0.0:
            phi = np.deg2rad(phi)
            s, c = np.sin(phi), np.cos(phi)
            A = np.float32([[c,-s], [ s, c]])
            corners = [[0, 0], [w, 0], [w, h], [0, h]]
            tcorners = np.int32( np.dot(corners, A.T) )
            x, y, w, h = cv2.boundingRect(tcorners.reshape(1,-1,2))
            A = np.hstack([A, [[-x], [-y]]])
            img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if tilt != 1.0:
            s = 0.8*np.sqrt(tilt*tilt-1)
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
            img = cv2.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
            A[0] /= tilt
        if phi != 0.0 or tilt != 1.0:
            h, w = img.shape[:2]
            mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
        Ai = cv2.invertAffineTransform(A)
        return img, mask, Ai
        
    def affine_detect(self,detector, img, mask=None, pool=None):
        '''
        affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs
        Apply a set of affine transormations to the image, detect keypoints and
        reproject them into initial image coordinates.
        See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.
        ThreadPool object may be passed to speedup the computation.
        '''
        params = [(1.0, 0.0)]
        for t in 2**(0.5*np.arange(1,6)):
            for phi in np.arange(0, 180, 72.0 / t):
                params.append((t, phi))
    
        def f(p):
            t, phi = p
            timg, tmask, Ai = self.affine_skew(t, phi, img)
            keypoints, descrs = detector.detectAndCompute(timg, tmask)
            for kp in keypoints:
                x, y = kp.pt
                kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
            if descrs is None:
                descrs = []
            return keypoints, descrs
    
        keypoints, descrs = [], []
        if pool is None:
            ires = it.imap(f, params)
        else:
            ires = pool.imap(f, params)
    
        for i, (k, d) in enumerate(ires):
            print 'affine sampling: %d / %d\r' % (i+1, len(params)),
            keypoints.extend(k)
            descrs.extend(d)
    
        print
        return keypoints, np.array(descrs)       
        
        
    def explore_match(self, win, img1, img2, kp_pairs, noMatches, status = None, H = None):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2] = img2
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        if noMatches is True:
            return win,vis
    
        if H is not None:
            corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
            corners = np.int32( cv2.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
            cv2.polylines(vis, [corners], True, (255, 255, 255))
        else:
            return win,vis
    
        if status is None:
            status = np.ones(len(kp_pairs), np.bool_)            
        p1 = np.int32([kpp[0].pt for kpp in kp_pairs])        

        p2 = np.int32([kpp[1].pt for kpp in kp_pairs]) + (w1, 0)
    
        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        kp_color = (51, 103, 236)
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                col = green
                cv2.circle(vis, (x1, y1), 2, col, -1)
                cv2.circle(vis, (x2, y2), 2, col, -1)
            else:
                col = red
                r = 2
                thickness = 3
                cv2.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
                cv2.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
                cv2.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
                cv2.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
        vis0 = vis.copy()
        for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
            if inlier:
                cv2.line(vis, (x1, y1), (x2, y2), green)
        return win, vis 
        
    def filter_matches(self,kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs