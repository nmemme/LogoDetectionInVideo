import numpy as np
import math
import collections
import os
import cv2

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


def logo_des(path):
    sift = cv2.xfeatures2d.SIFT_create()
    os.chdir(path)
    b = os.listdir(".")
    kp1 = []
    des1 = []
    a = b[1:len(b)-1]
    for i in a:
        img = cv2.imread(i, 0)
        # des is a matrix with 33 key points * 128 feature vectors,kp contains x & y
        kp, des = sift.detectAndCompute(img, None)
        kp1.append(kp)
        des1.append(des)
    os.chdir("..")
    return kp1, des1


if __name__ == '__main__':

    # read in whole dataset here !
    # use sift to extract kp & des
    sift = cv2.xfeatures2d.SIFT_create()
    path = "/Users/wangziji/Documents/Pycharm/EC500/sprint4/DHL1"

    kp1, des1 = logo_des(path)
    cap = cv2.VideoCapture('DHL.mp4')

    count = 0
    # roiBox = None
    roiBox = []
    roiHistt = []

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 360))
    # use brute-force knn to match the kps
    bf = cv2.BFMatcher()

    while( True ):
        # Capture frame-by-frame, each frame is 1080 pixels
        ret, frame = cap.read()
        orig = frame
        # bar = np.array([50,50,50,50])
        flag1 = 0
        # update ROI every 5 frames
        roiBox = []
        Box = []

        pic = frame
        gray_trg = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        kp2, des2 = sift.detectAndCompute(gray_trg, None)
        if des2 is not None:
            os.chdir(path)
            a = os.listdir(".")
            a = a[1:-1]
            index1 = 0
            for i in a:
                img = cv2.imread(i, 0)
                matches = bf.knnMatch(des1[index1], des2, k=2)
                if matches is not None:
                    kp_trg = bf_knnmatches(matches, img, kp1[index1], kp2)
                    # if not detecting logo in the image, just skip the tracking and show the original frame.
                    if len(kp_trg) >= 4:
                        roiPts = np.array(kp_trg)
                        s = roiPts.sum(axis=1)
                        tl, ld, ru, br = extrct_ROI(s)
                        # roiBox .append([tl[0], tl[1], br[0], br[1]])
                        roiBox = (tl[0], tl[1], br[0], br[1])
                        # if min(roiBox) > 0 and len(roiBox) > 0:
                        if min([tl, ld, ru, br]) > 0 and len([tl, ld, ru, br]) > 0:
                            Box.append([np.int32([ld, tl, ru, br])])
                            print index1
                print index1
                index1 += 1
        os.chdir("..")

        for k in range(len(Box)):
            # draw the rectangle
            # frame = cv2.polylines(frame, Box[k], True, 255, 3, cv2.LINE_AA)
            # blur the image
            x = Box[k][0][0][0]
            y = Box[k][0][0][1]
            x1 = Box[k][0][2][0]
            y1 = Box[k][0][2][1]
            sub = frame[y:y1, x:x1]
            sub = cv2.GaussianBlur(sub, (51, 51), 0, 0) # kernel length
            # sub = cv2.medianBlur(sub, 51)
            frame[y:y1, x:x1] = sub
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # write the frame to output
        out.write(frame)

        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        count += 1

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()






