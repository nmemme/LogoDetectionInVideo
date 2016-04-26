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
#import Image



# extract x & y of kp
def extract_position(kp):
    x=[]
    y=[]
    for i in range(len(kp)):
        x.append(kp[i][0])
        y.append(kp[i][1])
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
        if sort_dis[best_match_trgindex[k]] > 100:
            best_match_trgindex.remove(best_match_trgindex[k])
            k -= 1
        k += 1

    best_match = []
    kp_trg = []
    if len(best_match_trgindex) < 4:
        return match, minidis, kp_trg
    else:
        for i in best_match_trgindex:
            best_match.append([i, match[i]])
            kp_trg.append(kp2[i])
        return match, minidis, kp_trg

def extrct_ROI(kp):
    x, y = extract_position(kp)
    print "xxxxxx", x
    print "yyyyyy", y
    leftup = [int(math.floor(min(x))), int(math.floor(max(y)))]
    leftdown = [int(math.floor(min(x))), int(math.ceil(min(y)))]
    rightup = [int(math.ceil(max(x))), int(math.floor(max(y)))]
    rightdown = [int(math.ceil(max(x))), int(math.ceil(min(y)))]
    return [leftup, leftdown, rightup, rightdown]

def bf_knnmatches( matches, img):
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    good = []

    print "mmmmmmmmm", len(matches[0])
    '''
    print "m111111111", matches[0][0]
    print "m222222222", matches[0][1]
    '''
    dst = []
    if len(matches[0]) == 2:
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                print "ddddddddd", m.distance
                good.append(m)
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = img.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

                dst = cv2.perspectiveTransform(pts, M)
            else:
                dst = []
    else:
        dst = []
    return dst



if __name__ == '__main__':

    # Should read in whole dataset here!!!!

    # find the interest pts
    img = cv2.imread('2.png', 0)
    # use sift to extract kp & des
    sift = cv2.xfeatures2d.SIFT_create()
    # use brute-force knn to match the kps
    bf = cv2.BFMatcher()
    # des is a matrix with 33 key points * 128 feature vectors,kp contains x & y
    # kp1, des1 = sift.detectAndCompute(gray_logo, None)
    kp1, des1 = sift.detectAndCompute(img, None)


    cap = cv2.VideoCapture('DHL.mp4')
    count = 0
    roiBox = None

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 360))


    while( True ):
        # Capture frame-by-frame, each frame is 1080 pixels
        ret, frame = cap.read()
        orig = frame
        # print frame.shape

        # update ROI every 5 frames
        if count % 3 == 0:
            pic = frame
            '''
            for i in range(3):
                lower = cv2.pyrDown(pic)
                pic = lower
            '''
            gray_trg = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            kp2, des2 = sift.detectAndCompute(gray_trg, None)
            if des2 is not None:
                matches = bf.knnMatch(des1, des2, k=2)
                if matches is not None:
                    # print "matches", matches
                    kp_trg = bf_knnmatches(matches, img)

                    # if not detecting logo in the image, just skip the tracking and show the original frame.
                    if len(kp_trg) >= 4:
                        roiPts = np.array(kp_trg)
                        s = roiPts.sum(axis=1)
                        print "kkkp", kp_trg
                        print "ssssss", s
                        tl, ld, ru, br = extrct_ROI(s)
                        print "tl", tl
                        print "br", br
                        roi = orig[tl[1]:br[1], tl[0]:br[0]]
                        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                        roiBox = (tl[0], tl[1], br[0], br[1])
                        # initialize the termination criteria for cam shift, indicating
                        # a maximum of ten iterations or movement by a least one pixel
                        # along with the bounding box of the ROI
                        termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                        # print "jajajja",roiBox



        if roiBox is not None:
            # print "dsdfsdfs"
            # convert the current frame to the HSV color space
            # and perform mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            # apply cam shift to the back projection, convert the
            # points to a bounding box, and then draw them
            print "roiBox", roiBox
            print "bbbbbb", backProj
            print "ttttttt", termination
            if min(roiBox) > 0:
                (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
                pts = np.int0(cv2.boxPoints(r))
                frame = cv2.polylines(frame, [np.int32(kp_trg)], True, 255, 3, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # write the flipped frame
        out.write(frame)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()








