import cv2
import os
import numpy as np
from sklearn import svm
import random

def train():
	training_set=[]
	training_labels=[]
	os.chdir("/Users/muyunyan/Desktop/EC500FINAL/logo/")
	counter=0
	a=os.listdir(".")
	for i in a:
	 os.chdir(i)
	 print(i)
	 for d in os.listdir("."):
		 img = cv2.imread(d)
		 res=cv2.resize(img,(250,250))
		 gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		 xarr=np.squeeze(np.array(gray_image).astype(np.float32))
		 m,v=cv2.PCACompute(xarr)
		 arr= np.array(v)
		 flat_arr= arr.ravel()
		 training_set.append(flat_arr)
		 training_labels.append(i)
	 os.chdir("..")
	 trainData=training_set
	 responses=training_labels
	 svm = svm.SVC()
	 svm.fit(trainData,responses)
	 return svm

def test(path):
	cap = cv2.VideoCapture(path_video)
	testing=[]
	while(True):
		ret, frame = cap.read()
		res=cv2.resize(frame,(250,250))
		
		gray_image = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
		xarr=np.squeeze(np.array(gray_image).astype(np.float32))
		m,v=cv2.PCACompute(xarr)
		arr= np.array(v)
		flat_arr= arr.ravel()
		testing.append(flat_arr)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord("q"):
            break
	cap.release()
    cv2.destroyAllWindows()
	logos=svm.predict(testing)
	uniqlogos=list(set(logos))
	for i in uniqlogos:
		print(i)

path_video = '/Users/muyunyan/Desktop/EC500FINAL/EC500hey.mp4'
svm=train()
test(path_video)
