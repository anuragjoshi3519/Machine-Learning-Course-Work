import numpy
class SVC:
    def __init__(self,C=1.0):
        self.C=C
        self.W_=0
        self.b_=0
        
    def hingeLoss(self,X,Y,W,b):
        loss=0.5*numpy.dot(W,W.T)
        
        m=X.shape[0]
        
        for i in range(m):
            ti=Y[i]*(numpy.dot(W,X[i].T)+b)
            loss+=self.C*max(0,1-ti)
            
        return loss[0][0]
    
    def fit(self,X,Y,batch_size=120,learning_rate=0.001,max_itr=400):
        n=X.shape[1] # no. of features
        m=X.shape[0] # no. of samplesimage_data,labels
        
        W=numpy.zeros((1,n))
        b=0
        
        #training
        losses=[]
        
        for _ in range(max_itr):
            
            l=self.hingeLoss(X,Y,W,b)
            losses.append(l)
            
            #ids for mini batch
            ids=numpy.arange(m)
            numpy.random.shuffle(ids)
             
            #mini-batch gradient descent
            for batch_start in range(0,m,batch_size):
                gradw=0
                gradb=0
                for j in range(batch_start,batch_start+batch_size):
                    if j<m:
                        i=ids[j]
                        ti=Y[i]*(numpy.dot(W,X[i].T)+b)

                        if ti>1:
                            gradw+=0
                            gradb+=0
                        else:
                            gradw+=self.C*X[i]*Y[i]
                            gradb+=self.C*Y[i]
                
                W= W - learning_rate*(W - gradw)
                b= b + learning_rate*gradb
            
        self.W_=W
        self.b_=b
            
        return self.W_,self.b_,losses