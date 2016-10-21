from util import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    image = cv2.imread('/home/ply/lena.jpg',0);
    h, w = image.shape[:2]
 #   imgL = cv2.pyrDown(image)

  #  img = cv2.imread('/home/ply/lena.jpg',0);
   # h, w = img.shape[:2]
    img = np.zeros([h/2, w / 2])
    img2 = np.zeros([h/2,w/2])
    img1 = np.zeros([h/2,w/2])

    print h, w
    print h/2

    for j in xrange(h/2):
        for i in xrange(w/2):

            img[j, i] = image[2*j+1, 2 * i + 1]/4+image[2*j, 2 * i ] /4+image[2*j+1, 2 * i ]/4+image[2*j, 2 * i + 1]/4
            img1[j, i] = image[2*j+1, 2 * i + 1]
            img2[j, i] = image[2*j, 2 * i ]

        if j == 1:
            print  img[j, i],img1[j, i],img2[j, i]


    cv2.imwrite("/home/ply/chouyang.jpg", img);
    cv2.imwrite("/home/ply/chouyang1.jpg", img1);
    cv2.imwrite("/home/ply/chouyang2.jpg", img2);


