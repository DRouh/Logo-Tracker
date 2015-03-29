import numpy as np
import cv2
import math

def GetBoundingRectangle(blurred_im_hsv, h):    
    
    h_min = h - 10 if h - 10 > 0 else 0
    
    s_min = 100
    v_min = 100
    
    h_max = min(179, h + 10)
    s_max = 255
    v_max = 255

#    COLOR_MIN = np.array([math.floor(179 * h_min / 359), math.floor(255 * s_min / 100) , math.floor(255 * v_min / 100)])
#    COLOR_MAX = np.array([math.floor(179 * h_max / 359), math.floor(255 * s_max / 100 ), math.floor(255 * v_max / 100)])    

    
    COLOR_MIN = np.array([h_min, s_min, v_min])
    COLOR_MAX = np.array([h_max, s_max, v_max])
    
    frame_threshed = cv2.inRange(blurred_im_hsv, COLOR_MIN, COLOR_MAX)    
    frame_threshed = cv2.medianBlur(frame_threshed, 7)
        
    contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        return 0,0,0,0
    print areas[0]            
    max_index = np.argmax(areas)
    cnt=contours[max_index]
    
    x,y,w,h = cv2.boundingRect(cnt)    
    return x, y, x+w, y+h
    
    
def GetContours(blurred_im_hsv, h):    
    
    h_min = h - 10 if h - 10 > 0 else 0
    
    s_min = 100
    v_min = 100
    
    h_max = min(179, h + 10)
    s_max = 255
    v_max = 255

#    COLOR_MIN = np.array([math.floor(179 * h_min / 359), math.floor(255 * s_min / 100) , math.floor(255 * v_min / 100)])
#    COLOR_MAX = np.array([math.floor(179 * h_max / 359), math.floor(255 * s_max / 100 ), math.floor(255 * v_max / 100)])    

    
    COLOR_MIN = np.array([h_min, s_min, v_min])
    COLOR_MAX = np.array([h_max, s_max, v_max])
    
    frame_threshed = cv2.inRange(blurred_im_hsv, COLOR_MIN, COLOR_MAX)    
    frame_threshed = cv2.medianBlur(frame_threshed, 7)
        
    contours, hierarchy = cv2.findContours(frame_threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        return contours
            
    max_index = np.argmax(areas)
    cnt=contours[max_index]
 
    return cnt