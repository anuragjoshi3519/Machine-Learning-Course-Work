import cv2
import numpy as np
import os
#############################
#### Data Preprocessing #####

trainX=[]        # Image data
trainY=[]        # Labels for data
dataset_path='Data/'
classLabel=0
nameDict={}
for file in os.listdir(dataset_path):
    if file.endswith('.npy'):
        dataX=np.load(dataset_path+file)
        dataY=np.ones((dataX.shape[0],))*classLabel
        trainX.append(dataX)
        trainY.append(dataY)
        nameDict[classLabel]=file[:-4]
        classLabel+=1

trainX=np.concatenate(trainX,axis=0)
trainY=np.concatenate(trainY,axis=0)

############################
########## **K-NN** ########

def distance(xQuery,x):
    return np.sqrt(sum((x-xQuery)**2))

def knn(xQuery,X,Y,k):
    distVector=[]
    for i in range(X.shape[0]):
        d=distance(xQuery,X[i])
        distVector.append((d,Y[i]))
    
    distVector=sorted(distVector)
    distVector=np.array(distVector)
    distVector=distVector[:k]
    distVector=distVector[:,1]
    
    U=np.unique(distVector,return_counts=True)
    index=U[1].argmax()
    return U[0][index]

#############################
###### Face Prediction ######

cap = cv2.VideoCapture(0)
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    
    offset=8
    for face in faces:
        x,y,w,h=face
        inputImage=gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        inputImage=cv2.resize(inputImage,(100,100))
        inputImage=inputImage.reshape((-1,))
        label=knn(inputImage,trainX,trainY,4)
        cv2.rectangle(gray_frame,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,0),2)
        cv2.putText(gray_frame,nameDict[int(label)],(x-offset,y-offset-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2,cv2.LINE_AA)
         
    cv2.imshow("Predicting Names",gray_frame)
    key= cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
        



