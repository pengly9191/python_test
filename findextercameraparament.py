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
    from glob import glob

    img_mask = '/home/ply/pic/3.jpg'

    img_names = glob(img_mask)

    for fn in img_names:
        print 'processing %s...' % fn,
        img = cv2.imread(fn, 0)




    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(img, (columns, rows), flags=cv2.CALIB_CB_FAST_CHECK)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find corners with subpixel accuracy
    cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)


    cv2.drawChessboardCorners(img, (columns, rows), corners, ret)


    plot_image(img)
    plt.show()

