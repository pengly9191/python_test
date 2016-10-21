import cv2
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':

    ina = cv2.useOptimized()
    print ina

    img = cv2.imread('../../data/lena.jpg')
    cv2.setUseOptimized(False)
    e1 = cv2.getTickCount()
    res = cv2.medianBlur(img,49)
    e2 = cv2.getTickCount()
    print "time1:%s" % ((e2 - e1) / cv2.getTickFrequency())

