#!/usr/bin/env python

import cv2
import math
import pylab
import numpy as np
# local modules
from common import splitfn

# built-in modules
import os
import matplotlib.pyplot as plt

from util import *

if __name__ == '__main__':
    import sys
    import getopt


######### laser detection#############
    image = cv2.imread('/home/ply/pic/background  picture/pic-2.jpg',0)
    image_background = cv2.imread('/home/ply/pic/background  picture/pic-0.jpg',0)


    def r_rgb(image):
        return cv2.split(image)[0]


#    image = r_rgb(image)
#    image_background = r_rgb(image_background)

    image_diff_r = cv2.absdiff(image, image_background)

    cv2.imwrite('/home/ply/pic/background  picture/image_diff_r.jpg', image_diff_r)

    image_diff_r = cv2.GaussianBlur(image_diff_r, (15, 15), 12, 12)

    threshold_value = 20

 #   gray_img = getGray(image_diff_r)

  #  threshold_value = getThres(image_diff_r)




    image_blur_threshold = cv2.threshold(image_diff_r, threshold_value, 255, cv2.THRESH_BINARY)[1]

    image_blur = cv2.GaussianBlur(image_blur_threshold, (15, 15), 12, 12)

    image_blur_threshold = cv2.threshold(image_blur, threshold_value, 255, cv2.THRESH_BINARY)[1]

    # Compute ROI

    _,contours0, hierarchy = cv2.findContours(image_blur_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]

    cv2.drawContours(image_blur_threshold, contours, -1, (255, 255, 255), 3)

    area = []
    for i in xrange(len(contours)):
        # area = {}
        # area[i]=(cv2.contourArea(contours[i]))
        area.append(cv2.contourArea(contours[i]))
    area = np.array(area)
    print area
    cnt = area.argmax(axis=0)


    x, y, w, h = cv2.boundingRect(contours[cnt])
    img = cv2.rectangle(image_blur_threshold, (x, y), (x+w, y+h), (255, 255, 255), 2)

    img_h, img_w = img.shape[:2]
    roi_mask = np.zeros((img_h, img_w), np.uint8)
    p1 = (x,y)
    p2 = (x+w,y)
    p3 = (x,y+h)
    p4 = (x+w,y+h)
    points = np.array([p1, p2, p4, p3])
    cv2.fillConvexPoly(roi_mask, points, 255)

    cv2.imwrite('/home/ply/pic/background  picture/roi_mask.jpg',roi_mask)

