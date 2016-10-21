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


class LinearLeastSquares2D(object):

    def fit(self, data):
        data_mean = data.mean(axis=0)
        x0, y0 = data_mean
        if data.shape[0] > 2:  # over determined
            u, v, w = np.linalg.svd(data - data_mean)
            print 'u: ',u
            print 'v:   ',v
            print 'w:   ',w
            vec = w[0]
            print 'vec: ',vec

            theta = math.atan2(vec[0], vec[1])
        elif data.shape[0] == 2:  # well determined
            theta = math.atan2(data[1, 0] - data[0, 0], data[1, 1] - data[0, 1])
        theta = (theta + math.pi * 5 / 2) % (2 * math.pi)
        d = x0 * math.sin(theta) + y0 * math.cos(theta)
        return d, theta

    def residuals(self, model, data):
        d, theta = model
        dfit = data[:, 0] * math.sin(theta) + data[:, 1] * math.cos(theta)
        return np.abs(d - dfit)

    def is_degenerate(self, sample):
        return False

def ransac(data, model_class, min_samples, threshold, max_trials=10):

    best_model = None
    best_inlier_num = 0
    best_inliers = None
    data_idx = np.arange(data.shape[0])
    for _ in xrange(max_trials):
        sample = data[np.random.randint(0, data.shape[0], 2)]
        if model_class.is_degenerate(sample):
            continue
        sample_model = model_class.fit(sample)
        sample_model_residua = model_class.residuals(sample_model, data)
        sample_model_inliers = data_idx[sample_model_residua < threshold]
        inlier_num = sample_model_inliers.shape[0]
        if inlier_num > best_inlier_num:
            best_inlier_num = inlier_num
            best_inliers = sample_model_inliers
    if best_inliers is not None:
        best_model = model_class.fit(data[best_inliers])
    return best_model, best_inliers




if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob
#
#     img_mask = '/home/ply/pic/3.jpg'
#
#     img_names = glob(img_mask)
#
#
# # find corners
#     square_width = 20
#     pattern_size = (11, 8)
#     pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
#     pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
#     pattern_points *= square_width
#
#     obj_points = []
#     img_points = []
#     h, w = 0, 0
#     for fn in img_names:
#         print 'processing %s...' % fn,
#         img = cv2.imread(fn, 0)
#         plot_image(img)
#         plt.show()
#
#         if img is None:
#             print "Failed to load", fn
#             continue
#
#         h, w = img.shape[:2]
#
#         found, corners = cv2.findChessboardCorners(img, pattern_size)
#
#         if found:
#             term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
#             cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
#             print corners
#
#             vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#             cv2.drawChessboardCorners(vis, pattern_size, corners, found)
#             path, name, ext = splitfn(fn)
#             cv2.imwrite('%s/%s_chess.bmp' % ('/home/ply/pic', name), vis)
#             plot_images((img, vis))
#             plt.show()
#
#         if not found:
#             print 'chessboard not found'
#             continue
#         img_points.append(corners.reshape(-1, 2))
#  #       obj_points.append(pattern_points)
#
#
#         print 'ok'
#
#         camera_matrix = np.array([[1388, 0, 781], [0, 1380, 628], [0, 0, 1]])
#         distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
#
#         ret, rvecs, tvecs = cv2.solvePnP(pattern_points, corners, camera_matrix, distortion_coefficients)
#
#         if ret:
#             R = cv2.Rodrigues(rvecs)[0]
#             t = tvecs.T[0]
#             n = R.T[2]
#             d = np.dot(n, t)
#             print("\nRotation matrix\n\n{0}\n".format(R))
#             print("\nTranslation vector\n\n{0} mm\n".format(t))
#             print("\nPlane normal\n\n{0}\n".format(n))
#             print("\nPlane distance\n\n{0} mm\n".format(d))

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    tga = 150.0/330.0;
    l = 330.0;
    fx = 1/1388.35813;
    u0 = 781.440857;
    fy = 1/1380.78149;
    v0 = 628.483912;

    image = cv2.imread('/home/ply/pic/33.jpeg', 0);
    h, w = image.shape[:2]
    print h, w

    weight_matrix = np.array((np.matrix(np.linspace(0, w - 1, w)).T * np.matrix(np.ones(h))).T)

    # Compute center of mass

    s = image.sum(axis=1)
    v = np.where(s > 0)[0]
    u = (weight_matrix * image).sum(axis=1)[v] / s[v]

    print u
    print v

    Xw = ((1/fx)*(u-u0)*l*tga)/((1/fx)*(u-u0)+tga);
    Yw = -((1/fy) * (v - v0) * l * tga) / ((1/fx) * (u - u0) + tga);
    Zw = ((1/fx) * (u - u0) * l ) / ((1/fx) * (u - u0) + tga);

    print

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(Xw, Yw, Zw, rstride=1, cstride=1,cmap=cm.jet,
        linewidth=0, antialiased=False)


    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


    for i in xrange(100):
        print 'u: ', u[i], v[i];
        print 'Xw: ',Xw[i],Yw[i],Zw[i];








#
# ############  ROI  #############
#     roi_mask, image_gray = compute_ROI(corners, img)
# #    plot_images((roi_mask,image_gray))
#
#
# ######### laser detection#############
#     image = cv2.imread('../../data1/3.jpg', 0)
#     image_background = cv2.imread('../../data1/4.jpg', 0)
#
# ####  Monochromatic space  ###############
#
#     def r_rgb(image):
#         return cv2.split(image)[0]
#
#     image_r = r_rgb(image)
#     image_background_r = r_rgb(image_background)
#     image_diff_r = cv2.subtract(image_r, image_background_r)
#
#     # Hack to filter one laser (only for the example)
#     zeros = np.zeros((1280, 480), dtype=np.uint8)
#     ones = 255 * np.ones((1280, 480), dtype=np.uint8)
#     mask = np.concatenate((ones, zeros), axis=1)  # (zeros, ones)    one half zeros and one half ones
#     image_diff_r = cv2.bitwise_and(image_diff_r, mask)     #get one line
#
#     # ROI mask      !!!!!!!!!must notice the laser line is in roi_mask
#     image_diff_r = cv2.bitwise_and(image_diff_r, roi_mask)
#
#     # for i in xrange(1280):
#     #     for j in xrange(960):
#     #         image_diff_r[i, j] = image_diff_r[i, ((image_diff_r.shape[1]) - j - 1)]
#     # cv2.imwrite('../../data1/image_diff_ror.jpg', image_diff_r)
#
#     # plot_images((image_r,image_background_r,image_diff_r))
#     # plt.show()
#
# ########## threshold  ###############
#
#     image_threshold,image_blur_threshold = threshold_img(image_diff_r,threshold_value,blur_value)
#
# ########## Valid window mask  ###############
#
#
#     mask,image_stripe = wind_mask(image_blur_threshold,image_diff_r,window)
#
#     # plot_images((mask, image_stripe))
#
# ########## Peak detection  ###############
#
#
#     v, peaks = peak_det(image_stripe)
#
#     plt.subplot(2,2,1)
#     plt.rcParams['figure.figsize'] = (15, 3)
#     plt.plot(v, peaks, '.')
#     plt.ylim(305, 335)
#
# ##########    Center  of mass  ###############
#
#     v,u = center_mass_svd(image_stripe)
#
#     plt.subplot(2, 2, 2)
#
#
#     plt.rcParams['figure.figsize'] = (15, 3)
#
#     plt.plot(v, u, '--')
#     plt.ylim(305, 335)
#     plt.show()
#
#
#
# ##############################      RANSAC     ###############################################
#
#
#
#     data = np.vstack((v.ravel(), u.ravel())).T
#     model, inliers = ransac(data, LinearLeastSquares2D(), 2, 2)
#
#     dr, thetar = model
#     f = (dr - v * math.sin(thetar)) / math.cos(thetar)
#
#     ds, thetas = LinearLeastSquares2D().fit(data)
#     lr = (ds - v * math.sin(thetas)) / math.cos(thetas)
#
#     plt.subplot(2, 2, 3)
#     plt.rcParams['figure.figsize'] = (15, 5)
#     plt.plot(data[:, 0], data[:, 1], '.r', label='outliers')
#     plt.plot(data[inliers][:, 0], data[inliers][:, 1], '.b', label='inliers')
#
#     plt.subplot(2, 2, 4)
#     plt.plot(v, lr, '-r', label='least-squares solution of all points')
#     plt.plot(v, f, '-b', label='RANSAC solution')
#     plt.ylim(305, 335)
#
#     plt.legend(loc=1)
# #    plt.show()
#
#
# ########################### show image ################################
#
#
#
#     image_line_cm = np.zeros_like(image_threshold)
#     image_line_lr = np.zeros_like(image_threshold)
#
#     image_line_cm[v, np.around(u).astype(int)] = 255
#     image_line_lr[v, np.around(f).astype(int)] = 255
#
#     image_cm = cv2.merge((cv2.add(image_diff_r, image_line_cm), image_line_cm, image_line_cm))
#     image_lr = cv2.merge((cv2.add(image_diff_r, image_line_lr), image_line_lr, image_line_lr))
#
#     total()
#
#
#
#     plot_images((image_cm, image_lr))