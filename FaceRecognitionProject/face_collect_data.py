import cv2
import numpy as np
cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
data_path='Data/'
imageData=[]
file_name=input("Enter the name of the person: ")
stop=1
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    cv2.imshow('Frame',frame)
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f: f[2]*f[3])
    
    if len(faces)==0:
        continue
        
    for x,y,w,h in faces[-1:]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    
    # Taking frames into the imageData file
    offset=8
    frame_crop=frame[y-offset:y+h+offset,x-offset:x+w+offset]
    frame_crop=cv2.resize(frame_crop,(100,100))
    frame_crop=cv2.cvtColor(frame_crop,cv2.COLOR_BGR2GRAY)
    
    if stop%5==0:
        imageData.append(frame_crop)
        print(len(imageData))
    stop+=1
    
    
    cv2.imshow('Face',frame_crop)
    
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break

imageData=np.asarray(imageData)   
imageData=imageData.reshape((imageData.shape[0],-1))
print(imageData.shape)
np.save(data_path+file_name+'.npy',imageData)
print("Data saved Successfully")

cap.release()
cv2.destroyAllWindows()
