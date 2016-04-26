import cv2
import os
import numpy as np
from sklearn import svm
import random
training_set=[]
training_labels=[]
os.chdir("FlickrLogos-v2/classes/jpg")
counter=0
a=['dhl','starbucks','no-logo']
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
     if(i=='no-logo' and counter==50):
         break
     #if(counter==5):
 os.chdir("..")
 counter=0
         #break
       

trainData=training_set
responses=training_labels
svm = svm.SVC()
svm.fit(trainData,responses)
os.chdir("c:/Python27/DHL")
x=['dhl.jpg','dhl2.jpg','starbucks.png','starbucks2.png']
#os.chdir("FlickrLogos-v2/classes/jpg/apple")
testing=[]
for i in x:
    print(i)
    img = cv2.imread(i)

    res=cv2.resize(img,(250,250))

    gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    xarr=np.squeeze(np.array(gray_image).astype(np.float32))
    m,v=cv2.PCACompute(xarr)
    arr= np.array(v)
    flat_arr= arr.ravel()

    testing.append(flat_arr)


print(svm.predict(testing))
