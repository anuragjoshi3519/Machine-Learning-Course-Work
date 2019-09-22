from sklearn.preprocessing import LabelEncoder
import MySVM
import numpy as np

class OneVsOneSVM:
    '''
    Parameters
    ----------
    C : float, optional (default=1.0)
    Penalty parameter C of the error term.
    
    max_iter : int, optional (default=6000)
    
    learning_rate : float, optional (default=0.00001)
    
    Callable Functions
    ------------------
    
    fit(X,Y) : take data as X, target_labels as Y as input and trains the SVM model
    
    score(X,Y) : take data as X, target_labels as Y as input and returns the accuracy of the model
    
    '''
    def __init__(self,C=1.0,max_iter=6000,learning_rate=0.00001):
        self.max_iter=max_iter
        self.learning_rate=learning_rate
        self.C=C
        self.svm_classifiers={}
        
    def generateClasswiseData(self,X,Y):
        data={}

        no_of_classes=len(np.unique(Y))
        no_of_samples=X.shape[0]

        for i in range(no_of_classes):
            data[i]=[]

        for i in range(no_of_samples):
            data[Y[i]].append(X[i])

        for k in range(no_of_classes):
            data[k]=np.array(data[k])

        return data


    def getPairData(self,d1,d2):

        l1=d1.shape[0]
        l2=d2.shape[0]
        data=np.zeros((l1+l2,d1.shape[1]))
        labels=np.zeros(l1+l2)

        data[:l1]=d1
        data[l1:]=d2

        labels[:l1]=1
        labels[l1:]=-1

        return data,labels

    def fit(self,X,Y):
        global le
        le=LabelEncoder()
        le.fit(Y)
        Y=le.transform(Y)
        
        data=self.generateClasswiseData(X,Y)
        svc=MySVM.SVC(self.C)
        for i in range(len(data)):
            self.svm_classifiers[i]={}
            for j in range(i+1,len(np.unique(Y))):
                x,y=self.getPairData(data[i],data[j])
                wts,b,losses=svc.fit(x,y, learning_rate=self.learning_rate, max_itr=self.max_iter)
                self.svm_classifiers[i][j]=(wts,b)      

    def predict(self,X):
        X=np.array(X)
        classes=len(self.svm_classifiers)
        count=np.zeros(classes,)
        for i in range(classes):
            for j in range(i+1,classes):
                W = self.svm_classifiers[i][j][0]
                b = self.svm_classifiers[i][j][1]
                if (np.dot(W,X.T)+ b)>=0:
                    count[i]+=1
                else:
                    count[j]+=1

        index=np.argmax(count)
        return le.inverse_transform([index])

    def score(self,X,Y):
        count=0
        for i in range(X.shape[0]):
            if Y[i]==self.predict(X[i]):
                count+=1

        return count/X.shape[0]