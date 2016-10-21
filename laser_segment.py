#!/usr/bin/env python
from util import *


def laser_segment(image_diff):
####  Monochromatic space  ###############

    def r_rgb(image):
        return cv2.split(image)[0]

    image_diff_r = r_rgb(image_diff)
#    cv2.imwrite('%s/image_diff_r.jpg' % path,image_diff_r)

########## threshold  ###############
    threshold_value = 80
    blur_value = 3

#    th = getThres(image_diff_r)

    image_blur = cv2.GaussianBlur(image_diff_r,(15,15),12,12)
    image_blur_threshold = cv2.threshold(image_blur, threshold_value, 255, cv2.THRESH_BINARY)[1]
 #   cv2.imwrite('%s/image_blur_threshold.jpg'% path, image_blur_threshold)
    image_stripe = cv2.bitwise_and(image_diff_r, image_blur_threshold)
#    cv2.imwrite('%s/image_stripe.jpg'%path, image_stripe)

##########    Center  of mass  ###############

    s,v,u = center_mass_svd(image_stripe)
    plt.subplot(2, 1, 2)
#    plt.rcParams['figure.figsize'] = (5, 3)
    plt.plot(v, u, '.')
 #   plt.ylim(305, 335)
    plt.xlim(0,5000)
#    plt.show()

############################gaussian filter ############################
    #Detect stripe segments
    i = 0
    seg = []
    segments = [s[_r] for _r in np.ma.clump_unmasked(np.ma.masked_equal(s, 0))]

    for segment in segments:
        j = len(segment)
        seg.append(u[i:i + j])
        i += j

    # Show segments
    pylab.rcParams['figure.figsize'] = (15, 3)

    from itertools import cycle

    cycol = cycle('bgrcmk').next

    i = 0
    for segment in segments:
        j = len(segment)
        plt.plot(v[i:i + j], u[i:i + j], '.', color=cycol())
        i += j

    # Segmented gaussian filter
    sigma = 2.0
    f = np.array([])
    for useg in seg:
        fseg = scipy.ndimage.gaussian_filter(useg, sigma=sigma)
        f = np.concatenate((f, fseg))



########################### show image ################################
    image = cv2.imread('/%s/pic-3.jpg' % path, 1)
    image_line_cm = np.zeros_like(image_blur_threshold)
    image_line_lr = np.zeros_like(image_blur_threshold)

    image_line_cm[v, np.around(u).astype(int)] = 255
    image_line_lr[v, np.around(f).astype(int)] = 255

    image_line_cm = cv2.bitwise_and(image_line_cm, image_blur_threshold)
    image_line_lr = cv2.bitwise_and(image_line_lr, image_blur_threshold)

    image_line_cm = cv2.merge(( image_line_cm, image_line_cm, image_line_cm))
    image_cm = cv2.add(image, image_line_cm)
    image_line_lr = cv2.merge(( image_line_lr, image_line_lr, image_line_lr))
    image_lr = cv2.add(image, image_line_lr)


  #  image_cm = cv2.merge((cv2.add(image, image_line_cm), image_line_lr, image_line_lr))
 #   image_lr = cv2.merge((cv2.add(image_diff_r, image_line_lr), image_line_lr, image_line_lr))

    return image_line_cm,image_cm,image_line_lr,image_lr



########################################################################################################

path = '/home/ply/image/image20160524/laser_off/'
path_save = '/home/ply/image/image20160524/get/'


if __name__ == '__main__':
    from glob import glob


######### laser detection#############
    img_mask = '/%s/*.jpeg' %path
    img_names = glob(img_mask)
    pic_num = (np.array(img_names)).shape[0]

    #
    # for fn in xrange((pic_num+1)/2):
    #
    #     image = cv2.imread('/%s/pic-%d.jpg' %(path,((fn)*2)),0)
    #     image_background = cv2.imread('/%s/pic-%d.jpg' %(path,(fn*2+1)),0)
    #
    #     image_diff = cv2.subtract(image, image_background)
    #
    #     image_line_cm, image_cm, image_line_lr, image_lr = laser_segment(image_diff)
    #
    #     cv2.imwrite('%s/image_line_cm-%d.jpg' % (path_save,(fn)), image_line_cm)
    #     # cv2.imwrite('%s/image_cm-%d.jpg' % (path_save,(fn)), image_cm)
    #     cv2.imwrite('%s/image_line_lr-%d.jpg' % (path_save,(fn)), image_line_lr)
    #     # cv2.imwrite('%s/image_lr-%d.jpg' % (path_save,(fn)), image_lr)
    #


    image = cv2.imread('/home/ply/image/image20160524/laser_off/1.jpeg', 0)


    image_line_cm, image_cm, image_line_lr, image_lr = laser_segment(image)
    cv2.imwrite("/home/ply/image/image20160524/get/1.jpeg", image_line_cm)


    print "finished"