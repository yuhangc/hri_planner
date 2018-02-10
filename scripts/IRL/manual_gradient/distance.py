#!/usr/bin/env python

import numpy as np


class EuclideanDist(object):
    def __init__(self):
        self.g = None
        self.H = None

    def compute(self, x, xr):
        """
        :param x: T x (|A|x|X|) matrix, position of all agents
        :param xr: T x |X| matrix, position of robot
        :return:
        """
        # compute the distances
        T, nX = xr.shape
        nA = x.shape[1] / nX

        x_diff = x - np.tile(xr, (1, nA))

        dist = np.zeros((T, nA), dtype=float)
        for a in range(nA):
            xa_diff = x_diff[:, a*nX:(a+1)*nX]
            dist[:, a] = np.sum(np.square(xa_diff), axis=1)

        # compute gradient for each time step
        self.g = 2.0 * x_diff

        # compute Hessian for each time step/agent
        hess_uint = 2.0 * np.eye(nX)
        self.H = np.tile(hess_uint, (T, nA))

        # return the computed distances
        return dist

    def grad(self):
        return self.g

    def hessian(self):
        return self.H
