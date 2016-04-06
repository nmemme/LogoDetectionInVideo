import cv2
import os
import numpy as np
from sklearn import svm
import random
training_set=[]
training_labels=[]
os.chdir("FlickrLogos-v2/classes/jpg")
counter=0
a=os.listdir(".")
for i in a:
 os.chdir(i)
 print(i)
 for d in os.listdir("."):
     print(counter,d)
     img = cv2.imread(d)
     res=cv2.resize(img,(250,250))
     gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
     xarr=np.squeeze(np.array(gray_image).astype(np.float32))
     m,v=cv2.PCACompute(xarr)
     arr= np.array(v)
     flat_arr= arr.ravel()
     training_set.append(flat_arr)
     training_labels.append(i)
     counter+=1
     if(counter==5):
         os.chdir("..")
         counter=0
         break
       

trainData=training_set
responses=training_labels
svm = svm.SVC()
svm.fit(trainData,responses)
os.chdir("c:/Python27")
x=['apple.png','google.jpg','ad.jpg']
#os.chdir("FlickrLogos-v2/classes/jpg/apple")
testing=[]
for i in x:

    img = cv2.imread(i)
    res=cv2.resize(img,(250,250))
    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    xarr=np.squeeze(np.array(gray_image).astype(np.float32))
    m,v=cv2.PCACompute(xarr)
    arr= np.array(v)
    flat_arr= arr.ravel()

    testing.append(flat_arr)

print(svm.predict(testing))
