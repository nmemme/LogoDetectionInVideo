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
import collections
from matplotlib import pyplot as plt
import pandas as pa




# extract x & y of kp
def extract_position(kp):
    x=[]
    y=[]
    for i in range(len(kp)):
        x.append(kp[i].pt[0])
        y.append(kp[i].pt[1])
    return x,y

def distance(x,y):
    dis = np.linalg.norm(x - y)
    return dis

def make_trainlabel(kp):
    trainlabel = []
    for j in range(len(kp)):
        trainlabel.append(j)
    return trainlabel

def knnmatch(kp1,kp2,des1,des2,num_range):
    disArray = np.array([[0 for s in range(len(kp1))] for s in range(len(kp2))])
    # print np.shape(disArray)
    for i in range(len(kp2)):
        for j in range(len(kp1)):
            dis = distance(des1[j], des2[i])
            disArray[i][j] = dis
    match = {}
    minidis = {}
    for i in range(len(kp2)):
        match[i] = disArray[i].argmin()
        minidis[i] = min(disArray[i])

    sort_dis = collections.OrderedDict(sorted(minidis.items(), key=lambda x: x[1]))
    best_match_trgindex = sort_dis.keys()[0:num_range]
    k = 0
    for i in range(num_range):
        if sort_dis[best_match_trgindex[k]] > 20:
            best_match_trgindex.remove(best_match_trgindex[k])
            k -= 1
        k += 1

    best_match = []
    kp_trg = []
    # represent the top 10 points that matches with logo img
    # each elements is [point in img_target, point in img_logo]
    if len(best_match_trgindex) < 4:
        return match, minidis, kp_trg
    else:
        for i in best_match_trgindex:
            best_match.append([i, match[i]])
            kp_trg.append(kp2[i])
        return match, minidis, kp_trg

def extrct_ROI(kp):
    x, y = extract_position(kp)
    leftup = [int(math.floor(min(x))), int(math.floor(max(y)))]
    leftdown = [int(math.floor(min(x))), int(math.ceil(min(y)))]
    rightup = [int(math.ceil(max(x))), int(math.floor(max(y)))]
    rightdown = [int(math.ceil(max(x))), int(math.ceil(min(y)))]
    return [leftup, leftdown, rightup, rightdown]


if __name__ == '__main__':

    # Should read in whole dataset here!!!!

    # find the interest pts
    img = cv2.imread('apple.png')
    gray_logo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # use sift to extract kp & des
    sift = cv2.xfeatures2d.SIFT_create()
    # des is a matrix with 33 key points * 128 feature vectors,kp contains x & y
    kp1, des1 = sift.detectAndCompute(gray_logo, None)


    cap = cv2.VideoCapture('test.mp4')
    count = 0


    while( True ):
        # Capture frame-by-frame, each frame is 1080 pixels
        ret, frame = cap.read()

        orig = frame
        # update ROI every 5 frames
        if count % 5 == 0:
            gray_trg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = sift.detectAndCompute(gray_trg, None)
            match, minidis, kp_trg = knnmatch(kp1, kp2, des1, des2, 10)
            # img_trg_kp = cv2.drawKeypoints(gray_trg, kp_trg, gray_trg, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # if not detecting logo in the image, just skip the tracking and show the original frame.
            if len(kp_trg) < 4:
                roiBox = None
                cv2.imshow('frame', frame)
                continue

                # how to continue with this?

            roiPts = extrct_ROI(kp_trg)
            # determine the top-left and bottom-right points
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis=1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]
            # grab the ROI for the bounding box and convert it
            # to the HSV color space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # compute a HSV histogram for the ROI and store the
            # bounding box
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])
            # initialize the termination criteria for cam shift, indicating
            # a maximum of ten iterations or movement by a least one pixel
            # along with the bounding box of the ROI
            termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        if roiBox is not None:
            # convert the current frame to the HSV color space
            # and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.boxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        print "hey"
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()








