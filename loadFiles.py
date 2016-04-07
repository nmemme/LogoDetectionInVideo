import cv2
import os
import numpy as np
impath="C:/Python27/FlickrLogos-v2/classes/jpg/"
maskpath="C:/Python27/FlickrLogos-v2/classes/mask/"

a=['no-logo','apple','google']
training_set=[]
training_label=[]
counter=0
for i in a:
 print(i)
 for d in os.listdir(impath+i):
     print(counter,d)
     img = cv2.imread(impath+i+'/'+d)
     res=cv2.resize(img,(500,500))
     gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
     xarr=np.squeeze(np.array(gray_image).astype(np.float32))
     m,v=cv2.PCACompute(xarr)
     arr= np.array(v)
     flat_arr= arr.ravel()
     training_set.append(flat_arr)
     training_labels.append(i)
     counter+=1
