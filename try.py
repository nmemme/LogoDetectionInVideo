import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import image
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import math



# deal to be gray version
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('Apple.png')
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
#plt.show()
#plt.title('grey picture_logo')

# find the interest pts
img2 = cv2.imread('Apple.png')
img_target = cv2.imread('macbook.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img2, None)
kp2, des2 = sift.detectAndCompute(img_target, None)

gray_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray_target)
#plt.title('grey picture_train_image')
#plt.show()

i = 0
x=[]
y=[]
for i in range(len(kp1)):
    x.append(kp1[i].pt[0])
    y.append(kp1[i].pt[1])

#plt.plot(x,y)
#plt.title('logo feature points')
#plt.show()

# draw keypoint on the picture,second one is the one with keypoints
# keypoints are the extreme largest point & des are its other 3 descriptions

img3 = cv2.drawKeypoints(img2, kp1, img2)
cv2.imshow('SIFT1',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

img5 = cv2.drawKeypoints(img_target, kp2, img_target)
cv2.imshow('SIFT3',img5)
cv2.waitKey(0)
cv2.destroyAllWindows()

img4 = cv2.drawKeypoints(gray2,kp1,img2,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT2',img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

img6 = cv2.drawKeypoints(gray_target,kp2,img_target,flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('SIFT4',img6)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print des1 = 33 * 128


#
def RFClassify(trainData,trainLabel,testData):
    rfClf=RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    rfClf.fit_transform(trainData, trainLabel)
    testlabel=rfClf.predict(testData)
    return testlabel

trainlabel = []
for j in range(len(kp1)):
    trainlabel.append(j)
# print trainlabel
trainData = des1
testData = des2
RF_label_predict = RFClassify(trainData, trainlabel, testData)
# print RF_label_predict

# Need to draw only good matches, so create a mask
matches = {}
num_2 = len(kp2)
for i in range(num_2):
    matches[i] = ([i,RF_label_predict[i]])

# matches = sp.sparse.coo_matrix(([1]*num_2,(A,RF_label_predict)),shape=(num_2,num_2))
matchesMask = [[0,0] for i in xrange(len(matches))]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img_mask = cv2.drawMatches(img2,kp1,img_target,kp2,matches[:10],flags=2)

plt.imshow(img_mask,)
plt.show()

'''
# draw matches with flann
#img5 = cv2.drawMatches(img2, kp1, img_target, kp2, matches1to2)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img_mask = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img_mask,)
plt.show()
'''