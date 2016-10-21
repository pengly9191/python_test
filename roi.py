from util import *

from common import splitfn
import numpy as np
import sys
import getopt
from glob import glob
from PIL import Image

if __name__ == '__main__':

    args, img_mask = getopt.getopt(sys.argv[1:], '', ['save=', 'debug=', 'square_size='])
    args = dict(args)

    img_mask = '/home/ply/image/20160720/out/*.jpg'
    img_names = glob(img_mask)

    for fn in img_names:
        print 'processing %s...' % fn
        img = cv2.imread(fn, 1)
        h, w = img.shape[:2]
        print h,w

        s = img.sum(axis=1)
        v = np.where(s > 0)[0]
        s = max(v)
        print s
        cutimg = np.zeros([s-20, w])
        cutimg = img[10:s-10,:]

        h,w = cutimg.shape[:2]
        print h,w

        cut1 = cutimg[10:h-10,10:w/2-40]
        cut2 = cutimg[10:h-10,w/2+10:w-40]
        h1,w1 = cut1.shape[:2]
        print h1,w1
        images = cv2.resize(img, (1191, 2*w1))
        image = images

        image[:,0:w1] = cut1
        image[:, w1:2*w1] = cut2

        path, name, ext = splitfn(fn)
        cv2.imwrite('/home/ply/image/20160720/cut-%s.jpg' % name, cut1)
        cv2.imwrite('/home/ply/image/20160720/cut-%s.jpg' % name, cut2)
        cv2.imwrite('/home/ply/image/20160720/image-%s.jpg' % name, image)


        merged2 = cv2.resize(cutimg, (2048, 1536))
        path, name, ext = splitfn(fn)
        cv2.imwrite('/home/ply/image/20160720/left-%s.jpg' % name, merged2)


