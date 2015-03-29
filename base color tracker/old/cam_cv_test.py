import cv2
import numpy as np
import boundingrectangle as br


cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()
    
    # here we dilate the image so we can better threshold the colors

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #blur image
    
    hsv = cv2.medianBlur(hsv, 7)
    
    #find three bounding rectangles
    rectangle0 = br.GetBoundingRectangle(hsv, 170)    
    t = br.GetContours(hsv, 170)
    cv2.drawContours(frame,t,-1,(0,255,0),3)
    rectangle1 = br.GetBoundingRectangle(hsv, 90)
    rectangle2 = br.GetBoundingRectangle(hsv, 160)
    
    h_min = 170#170
    s_min = 100
    v_min = 100
    
    h_max = 179
    s_max = 255
    v_max = 255

    
    COLOR_MIN = np.array([h_min, s_min, v_min])
    COLOR_MAX = np.array([h_max, s_max, v_max])    
    

    frame_threshed = cv2.inRange(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), COLOR_MIN, COLOR_MAX)  
    
    result = cv2.bitwise_and(frame, frame, mask = frame_threshed)
    #draw bounding rectangles
    cv2.rectangle(frame,(rectangle0[0],rectangle0[1]),(rectangle0[2],rectangle0[3]),(0, 255, 0),2)
    cv2.rectangle(frame,(rectangle1[0],rectangle1[1]),(rectangle1[2],rectangle1[3]),(0, 128, 128),2)
    cv2.rectangle(frame,(rectangle2[0],rectangle2[1]),(rectangle2[2],rectangle2[3]),(64, 64, 64),2)
      
        
    cv2.imshow('frame_threshed',frame_threshed)
    cv2.imshow('frame',frame)
    cv2.imshow('result', result)
    #cv2.imshow('frame',hsv)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    

cv2.destroyAllWindows()
cap.release()