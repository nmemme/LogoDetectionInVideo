import cv2
import numpy as np
import math
import collections
import os
import sys

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


def extrct_ROI(kp):
    x, y = extract_position(kp)
    leftup = [int(math.floor(min(x))), int(math.floor(max(y)))]
    leftdown = [int(math.floor(min(x))), int(math.ceil(min(y)))]
    rightup = [int(math.ceil(max(x))), int(math.floor(max(y)))]
    rightdown = [int(math.ceil(max(x))), int(math.ceil(min(y)))]
    return [leftup, leftdown, rightup, rightdown]

def bf_knnmatches( matches, img, kp1, kp2):
    MIN_MATCH_COUNT = 10
    # store all the good matches as per Lowe's ratio test.
    good = []
    dst = []
    if len(matches[0]) == 2:
        for m, n in matches:
            if m.distance < 0.7*n.distance:
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


def logo_des(path, kp1, des1):# path = images' folder
    sift = cv2.xfeatures2d.SIFT_create()
    os.chdir(path)
    a = os.listdir(".")
    #print b
    # a = b[0:len(b)]
    for i in a[1:len(a)]:
        print i
        img = cv2.imread(i, 0)
        # des is a matrix with 33 key points * 128 feature vectors,kp contains x & y
        kp, des = sift.detectAndCompute(img, None)
        kp1.append(kp)
        des1.append(des)
    os.chdir("..")
    return kp1, des1


if __name__ == '__main__':
    # use sift to extract kp & des
    sift = cv2.xfeatures2d.SIFT_create()
    # use brute-force knn to match the kps
    bf = cv2.BFMatcher()
    # whole folder with all classes of images
    # path_image = "/Users/muyunyan/Desktop/EC500FINAL/logo/"
    path_image = "/Users/muyunyan/Desktop/EC500FINAL/logo/"
    root = path_image
    # video path
    path_video = '/Users/muyunyan/Desktop/EC500FINAL/starbucks.mp4'
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (960, 540))

    # read in specific class of images and store sift features as matrix des1 & kp1
    # sys.argv[i] store the string name of class
    kp1 = []
    des1 = []
    os.chdir(root)
    logo_name = sys.argv[1: len(sys.argv)]
    for i in range(len(logo_name)):
        path = root + logo_name[i]
        # print path
        # os.chdir(path)
        # b = os.listdir(".")
        # for k in b[1:len(b)-1]:
        print path
        kp1, des1 = logo_des(path, kp1, des1)
        # print "llllll",len(des1)
        # kp1.append(kpp)
        # des1.append(dess)
        # os.chdir("..")
        # print des1
    # Now position back to root

    # read in every frame in video
    cap = cv2.VideoCapture(path_video)
    count = 0

    # roiBox = None
    roiBox = []
    roiHistt = []
    termination = []

    while(cap is not None):
        k = 0
        # Capture frame-by-frame, each frame is 1080 pixels
        ret, frame = cap.read()
        Box = []
        orig = frame
        gray_trg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray_trg, None)

        if count % 10 == 0:
            roiBox = []
            roiHistt = []
            termination = []

        if des is not None:
            os.chdir(path_image)
            a = os.listdir(".")
            a = a[1:len(a)-1]
            index1 = 0
            # image folder for each class
            for j in range(len(logo_name)):
                path = root + logo_name[j]
                os.chdir(path)
                b = os.listdir(".")
                # match each image in the folder with target
                for i in b[1:len(b)-1]:
                    img = cv2.imread(i, 0)
                    matches = bf.knnMatch(des1[index1], des, k=2)
                    if matches is not None:
                        kp_trg = bf_knnmatches(matches, img, kp1[index1], kp)
                        # if not detecting logo in the image, just skip the tracking and show the original frame.
                        if len(kp_trg) >= 4:
                            roiPts = np.array(kp_trg)
                            s = roiPts.sum(axis=1)
                            tl, ld, ru, br = extrct_ROI(s)

                            roiPts = np.array(kp_trg)
                            s = roiPts.sum(axis=1)
                            tl, ld, ru, br = extrct_ROI(s)
                            area = np.array([tl[0], tl[1], br[0], br[1]])
                            roi = orig[tl[1]:br[1], tl[0]:br[0]]
                            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
                            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
                            roiHistt.append(roiHist)
                            # roiBox .append([tl[0], tl[1], br[0], br[1]])
                            termination .append((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
                            if min([tl, ld, ru, br]) > 0 and len([tl, ld, ru, br]) > 0:
                                Box.append([np.int32([ld, tl, ru, br])])
                                # print index1
                    # print index1
                    index1 += 1
                os.chdir("..")
                # back to path_image
            os.chdir("..")
            # back to whole folder
        #print "llll", len(Box)

        print "bbbbbbb", Box


        if count % 10 == 0:
            roiBox = Box
            print Box, "heihei"
        if len(roiBox) > 0:
            for kk in range(len(roiBox)):
                # convert the current frame to the HSV color space
                # and perform mean shift
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                backProj = cv2.calcBackProject([hsv], [0], roiHistt[kk], [0, 180], 1)
                # apply cam shift to the back projection, convert the
                # points to a bounding box, and then draw them
                # print roiBox[kk][0]
                roiBox[kk] = tuple([roiBox[kk][0][1][0],roiBox[kk][0][1][1],roiBox[kk][0][3][0],roiBox[kk][0][3][1]])
                if (np.array(roiBox[kk])).all > 0:
                    print backProj, roiBox[kk], termination[kk]
                    (r, roiBox[kk]) = cv2.CamShift(backProj, roiBox[kk], termination[kk])
                    pts = np.int0(cv2.boxPoints(r))
                    ldd = [roiBox[kk][0], roiBox[kk][3]]
                    tll = [roiBox[kk][0], roiBox[kk][1]]
                    rur = [roiBox[kk][2], roiBox[kk][1]]
                    brr = [roiBox[kk][2], roiBox[kk][3]]
                    print [np.int32([ldd, tll, rur, brr])]
                    Box.append([np.int32([ldd, tll, rur, brr])])
                    print Box,"hoho"

        if len(Box) > 0:
            for k in range(len(Box)):
                # blur the image
                print "luuu",Box
                print "lalalla",Box[k][0]
                x = Box[k][0][0][0]
                y = Box[k][0][0][1]
                x1 = Box[k][0][2][0]
                y1 = Box[k][0][2][1]
                sub = frame[y:y1, x:x1]
                sub = cv2.GaussianBlur(sub, (51, 51), 0, 0) # kernel length
                frame[y:y1, x:x1] = sub
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # write the frame output
        out.write(frame)
        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count += 1
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()






