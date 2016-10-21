import matplotlib.pyplot as plt

from util import *


if __name__ == '__main__':
    import sys
    import getopt
    from glob import glob

    data = np.arange(10).reshape(5, 2)
    print 'data: ',data
    data_mean = data.mean(axis=0)
    print 'data_mean: ', data_mean
    x0, y0 = data_mean
    if data.shape[0] > 2:  # over determined
        u, v, w = np.linalg.svd(data - data_mean)
        print data-data_mean
        print 'u: ', u
        print 'v:   ', v
        print 'w:   ', w

        vec = w[0]
        print 'vec: ', vec

        theta = math.atan2(vec[0], vec[1])
    elif data.shape[0] == 2:  # well determined
        theta = math.atan2(data[1, 0] - data[0, 0], data[1, 1] - data[0, 1])
    theta = (theta + math.pi * 5 / 2) % (2 * math.pi)
    d = x0 * math.sin(theta) + y0 * math.cos(theta)
    print d
    print (d - 7 * math.cos(theta))/math.sin(theta)
