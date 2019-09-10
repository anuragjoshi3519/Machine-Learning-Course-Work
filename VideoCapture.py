#!/usr/bin/env python

import cv2
cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret==False:
        continue
    
    cv2.imshow('Webcam',frame)
      			
    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
    
    
