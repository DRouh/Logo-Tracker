import cv2
import cv2.cv as cv
 
cap = cv2.VideoCapture(0)
 
#set camera width and height
CAM_WIDTH = 640
CAM_HEIGHT = 480
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
 
selected_color = None
image_origin = None
 
 
def onmouse(event, x, y, flags, param):
    global image_origin, selected_color
    if flags & cv2.EVENT_FLAG_LBUTTON:
        #taking squire cur 4x4 and scale it to 1x1
        cut = image_origin[y-1:y+2, x-1:x+2]
        cut = cv2.pyrDown(cut)
        cut = cv2.pyrDown(cut)
 
        selected_color = (int(cut[0][0][0]), int(cut[0][0][1]), int(cut[0][0][2]))
        #conveting to HSV and printing result
        selected_color_HSV = cv2.cvtColor(cut, cv2.COLOR_BGR2HSV)
        print(selected_color_HSV)
 
while(1):
 
    # Take each frame
    _, frame = cap.read()
    image_origin = frame.copy()
 
 
    #drawing selected colors
    if selected_color is not None:
        cv2.circle(frame, (CAM_WIDTH-20,20), 20, selected_color, -1)
 
    #show image and set callback
    cv2.imshow('img', frame)
    cv2.setMouseCallback('img', onmouse)
 
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
 
cv2.destroyAllWindows()