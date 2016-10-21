#!/usr/bin/env python

from util import *

class PlaneDetection(object):
    def fit(self, X):
        M, Xm = self._compute_m(X)
        U = linalg.svds(M, k=2)[0]
        normal = np.cross(U.T[0], U.T[1])
        # normal = numpy.linalg.svd(M)[0][:,2]
        if normal[2] < 0:
            normal *= -1
        dist = np.dot(normal, Xm)
        return dist, normal, M

    def residuals(self, model, X):
        _, normal, _ = model
        M, Xm = self._compute_m(X)
        return np.abs(np.dot(M.T, normal))

    def is_degenerate(self, sample):
        return False

    def _compute_m(self, X):
        n = X.shape[0]
        Xm = X.sum(axis=0) / n
        M = np.array(X - Xm).T
        return M, Xm


def ransac(data, model_class, min_samples, threshold, max_trials=500):
    best_model = None
    best_inlier_num = 0
    best_inliers = None
    data_idx = np.arange(data.shape[0])
    for _ in xrange(max_trials):
        sample = data[np.random.randint(0, data.shape[0], 3)]
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


    def compute_pc(name):
        # Load point cloud
        X = load_ply(name).vertexes


        n = X.shape[0]
        Xm = X.sum(axis=0) / n
        M = np.array(X - Xm).T

        # Equivalent to:
        #  numpy.linalg.svd(M)[0][:,2]
        # But 1200x times faster for large point clouds
        U = linalg.svds(M,k=2)[0]

        print U

        normal = np.cross(U.T[0], U.T[1])
        if normal[2] < 0:
            normal *= -1



        dist = np.dot(normal, Xm)
        std = np.dot(M.T, normal).std()

        print("\nNormal vector\n\n{0}\n".format(normal))
        print("\nPlane distance\n\n{0} mm\n".format(dist))
        print("\nStandard deviation\n\n{0} mm\n".format(std))


    compute_pc('/home/ply/laserpc.ply')


    def compute_ransac_pc(name):
        # Load point cloud
        X = load_ply(name).vertexes

        model, inliers = ransac(X, PlaneDetection(), 3, 0.1)

        dist, normal, M = model
        std = np.dot(M.T, normal).std()

        print("\nNormal vector\n\n{0}\n".format(normal))
        print("\nPlane distance\n\n{0} mm\n".format(dist))
        print("\nStandard deviation\n\n{0} mm\n".format(std))

    compute_ransac_pc('/home/ply/laserpc.ply')