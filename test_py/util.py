import cv2
import math
import pylab
import struct
import datetime
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import linalg



_begin = 0
total_time = datetime.timedelta()


################# GLOBAL VAR##############################
rows = 6
columns = 11
square_width = 13
threshold_value = 10
blur_value = 5

path = "../../data1/"


# Time measurement functions

def begin():
    global _begin
    _begin = datetime.datetime.now()


def end():
    global _begin, total_time
    end = datetime.datetime.now() - _begin
    total_time += end
    print('Time: %s' % end)


def total():
    global total_time
    print('Total time: %s' % total_time)


# Plot image functions

def plot_image(image):
    if len(image.shape) == 2:
        image = cv2.merge((image, image, image))
    f, axarr = plt.subplots(1, 1, figsize=(10, 15))
    axarr.axis('off')
    axarr.imshow(image)
    axarr.plot()


def plot_images(images):
    f, axarr = plt.subplots(1, len(images), figsize=(15, 15))
    for i in range(len(images)):
        if len(images[i].shape) == 2:
            image = cv2.merge((images[i], images[i], images[i]))
        else:
            image = images[i]
        axarr[i].axis('off')
        axarr[i].imshow(image)
        axarr[i].plot()


# Load image function

def load_image(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Pattern parameters



    #######################  COMPUTRE_ROI  ############################


def compute_ROI(corners,img):

    corners = corners.astype(np.int)

    # get four point
    p1 = corners[0][0]
    p2 = corners[columns - 1][0]
    p3 = corners[columns * (rows - 1)][0]
    p4 = corners[columns * rows - 1][0]

    # Compute ROI
    h, w = img.shape[:2]
    roi_mask = np.zeros((h, w), np.uint8)
    points = np.array([p1, p2, p4, p3])
    cv2.fillConvexPoly(roi_mask, points, 255)

    image_gray = cv2.bitwise_and(img, roi_mask)

    return roi_mask,image_gray


#######################  threshold_img  ############################

def threshold_img(image_diff_r,threshold_value,blur_value):


    image_threshold = cv2.threshold(image_diff_r, threshold_value, 255, cv2.THRESH_TOZERO)[1]

    # Blur image


    image_blur = cv2.blur(image_threshold, (blur_value, blur_value))

    image_blur_threshold = cv2.threshold(image_blur, threshold_value, 255, cv2.THRESH_TOZERO)[1]

    cv2.imwrite('../../data1/image_theshold.jpg',image_threshold)
    cv2.imwrite('../../data1/image_blur.jpg',image_blur)
    cv2.imwrite('../../data1/image_blur_threshold.jpg',image_blur_threshold)


#    plot_images((image_threshold, image_blur_threshold))

    return image_threshold,image_blur_threshold

#######################  valid window mask  ############################

def wind_mask(image_blur_threshold,image_diff_r,window):

    peak = image_blur_threshold.argmax(axis=1)  # the pos of every row max value

    _min = peak - window
    _max = peak + window

    mask = np.zeros_like(image_blur_threshold)

    for i in xrange(image_blur_threshold.shape[0]):
        mask[i, _min[i]:_max[i]] = 255


    image_stripe = cv2.bitwise_and(image_diff_r, mask)
    cv2.imwrite('../../data/mask.jpg', mask)
    cv2.imwrite('../../data/image_stripe.jpg',image_stripe)

    return mask,image_stripe


#######################  peak detection  ############################
# get max value
def peak_det(image_stripe):

    s = image_stripe.sum(axis=1)

    v = np.where(s > 0)[0]

    peaks = image_stripe.argmax(axis=1)[v]

    return v,peaks

# Pattern detection functions


#######################  svd fitting  ############################

def center_mass_svd(image_stripe):
    h, w = image_stripe.shape[:2]
    print h,w

    weight_matrix = np.array((np.matrix(np.linspace(0, w - 1, w)).T * np.matrix(np.ones(h))).T)


    # Compute center of mass

    s = image_stripe.sum(axis=1)
    v = np.where(s > 0)[0]
    u = (weight_matrix * image_stripe).sum(axis=1)[v] / s[v]



    # Show center of mass
    return s,v, u




def pattern_detection(image):
    # Convert image to 1 channel
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (columns, rows), flags=cv2.CALIB_CB_FAST_CHECK)

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Find corners with subpixel accuracy
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return corners, ret


def draw_pattern(image, corners, ret):
    # Draw corners into image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.drawChessboardCorners(image, (columns, rows), corners, ret)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# Read PLY functions

class Mesh(object):
    def __init__(self):
        self.vertexes = None
        self.colors = None
        self.normal = None
        self.vertex_count = 0


def _load_binary(mesh, stream, dtype, count):
    data = np.fromfile(stream, dtype=dtype, count=count)

    fields = dtype.fields
    mesh.vertex_count = count

    if 'v' in fields:
        mesh.vertexes = data['v']
    else:
        mesh.vertexes = np.zeros((count, 3))

    if 'n' in fields:
        mesh.normal = data['n']
    else:
        mesh.normal = np.zeros((count, 3))

    if 'c' in fields:
        mesh.colors = data['c']
    else:
        mesh.colors = 255 * np.ones((count, 3))


def load_ply(filename):
    m = Mesh()
    with open(filename, "rb") as f:
        dtype = []
        count = 0
        format = None
        line = None
        header = ''

        while line != 'end_header\n' and line != '':
            line = f.readline()
            header += line
        # Discart faces
        header = header.split('element face ')[0].split('\n')

        if header[0] == 'ply':

            for line in header:
                if 'format ' in line:
                    format = line.split(' ')[1]
                    break

            if format is not None:
                if format == 'ascii':
                    fm = ''
                elif format == 'binary_big_endian':
                    fm = '>'
                elif format == 'binary_little_endian':
                    fm = '<'

            df = {'float': fm + 'f', 'uchar': fm + 'B'}
            dt = {'x': 'v', 'nx': 'n', 'red': 'c', 'alpha': 'a'}
            ds = {'x': 3, 'nx': 3, 'red': 3, 'alpha': 1}

            for line in header:
                if 'element vertex ' in line:
                    count = int(line.split('element vertex ')[1])
                elif 'property ' in line:
                    props = line.split(' ')
                    if props[2] in dt.keys():
                        dtype = dtype + [(dt[props[2]], df[props[1]], (ds[props[2]],))]

            dtype = np.dtype(dtype)

            if format is not None:
                if format == 'binary_big_endian' or format == 'binary_little_endian':
                    _load_binary(m, f, dtype, count)
            return m
        else:
            return None

########################  otsu ########################################################


def getGray(img):
    numGray = [0 for i in range(pow(2, img.depth))]
    for h in range(img.height):
        for w in range(img.width):
            numGray[int(img[h, w])] += 1
    return numGray


def getThres(gray):
    maxV = 0
    bestTh = 0
    w = [0 for i in range(len(gray))]
    print w
    px = [0 for i in range(len(gray))]
    w[0] = gray[0]
    px[0] = 0
    for m in range(1, len(gray)):
        w[m] = w[m - 1] + gray[m]
        px[m] = px[m - 1] + gray[m] * m
    for th in range(len(gray)):
        w1 = w[th]
        w2 = w[len(gray) - 1] - w1
        if (w1 * w2 == 0):
            continue
        u1 = px[th] / w1
        u2 = (px[len(gray) - 1] - px[th]) / w2
        v = w1 * w2 * (u1 - u2) * (u1 - u2)
        if v > maxV:
            maxV = v
            bestTh = th
    return bestTh


def OtsuGray(grayImage):
    hist = cv2.CreateHist([256], cv2.CV_HIST_ARRAY, [[0, 256]])
    cv2.ClearHist(hist)

    cv2.CalcHist([grayImage], hist)


    totalH = 0
    for h in range(0, 256):
        v = cv2.QueryHistValue_1D(hist, h)
        if v == 0: continue
        totalH += v * h


    width = grayImage.width
    height = grayImage.height
    total = width * height


    v = 0

    gMax = 0.0

    tIndex = 0

    # temp
    n0Acc = 0
    n1Acc = 0
    n0H = 0
    n1H = 0
    for t in range(1, 255):
        v = cv2.QueryHistValue_1D(hist, t - 1)
        if v == 0: continue

        n0Acc += v
        n1Acc = total - n0Acc
        n0H += (t - 1) * v
        n1H = totalH - n0H

        if n0Acc > 0 and n1Acc > 0:
            u0 = n0H / n0Acc
            u1 = n1H / n1Acc
            w0 = n0Acc / total
            w1 = 1.0 - w0
            uD = u0 - u1
            g = w0 * w1 * uD * uD



            if gMax < g:
                gMax = g
                tIndex = t




    return tIndex