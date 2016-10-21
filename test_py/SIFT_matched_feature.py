import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *

print cv2.__version__


if __name__ == '__main__':
    img1= cv2.imread('/home/ply/image/20160721/left-pic-1.jpg',0);
    img2= cv2.imread('/home/ply/image/20160721/right-pic-1.jpg',0);

#    src1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#    src2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1,des1= sift.detectAndCompute(img1,None)
    kp2,des2= sift.detectAndCompute(img2,None)


    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    pts1 = []
    pts2 = []

    for i,(m, n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

 #   img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,None,flags=2)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F,mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)  #this flag  can be changed

    print F
    F=  np.int32(F)

    #we select only  inlier points
    # pts1 = pts1[mask.ravel()==1]
    # pts2 = pts2[mask.ravel()==1]
    h,w = img1.shape[:2]

    H1, H2 = cv2.stereoRectifyUncalibrated(pts1,pts2,F,(h,w),20)


    plt.figure(1)
 #   plot_image(img3)
 #   plt.show()


