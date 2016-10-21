import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':

    e1= cv2.getTickCount()

    img1 = cv2.imread('../../data/lena.jpg')
    img2 = cv2.imread('../../data/opencv-logo.png')

    if img1 is None or img2 is None:
        print 'read pic error '

    rows,cols,channels = img2.shape
    roi = img1[0:rows,0:cols]


    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret,mask = cv2.threshold(img2gray,175,255,cv2.THRESH_BINARY)
    mask_inv =  cv2.bitwise_not(mask)

    e2 = cv2.getTickCount()
   
    time = (e2-e1)/cv2.getTickFrequency()
    print time
    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

    dst = cv2.add(img1_bg,img2_fg)
    img1[0:rows,0:cols] = dst
    plt.imshow(img1)
    plt.show()
