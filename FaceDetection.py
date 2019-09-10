import cv2
cap=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('FaceRecognitionProject/haarcascade_frontalface_alt.xml')

while True:
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue

    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)    
    
    cv2.imshow('Webcam',frame)    			

    key = cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
    
