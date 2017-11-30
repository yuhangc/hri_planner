#!/usr/bin/env python

import numpy as np


class FeatureBase(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.f(args)

    def f(self, *args):
        raise Exception("method must be implemented by derived classes!")

    def grad(self, *args):
        raise Exception("method must be implemented by derived classes!")

    def hessian(self, *args):
        raise Exception("method must be implemented by derived classes!")


class Velocity(FeatureBase):
    def f(self, x0, u):
        """
        :param x0: 1xds numpy array 
        :param u: Txdu numpy array
        """
        return np.sum(np.square(u)) / float(len(u))

    def grad(self, x0, u):
        """ 
        :return: (Txdu)x1 vector of the gradient with respect to u 
        """
        return 2.0 * u.flatten() / float(len(u))

    def hessian(self, x0, u):
        """ 
        :return: (Txdu)x(Txdu) vector of the Hessian with respect to u 
        """
        return 2.0 * np.eye(u.size, dtype=float) / float(len(u))


class Acceleration(FeatureBase):
    def __init__(self, dt):
        super(Acceleration, self).__init__()
        self.dt = dt

    def f(self, x0, u):
        acc = np.diff(u, axis=0) / self.dt
        return np.sum(np.square(acc)) / float(len(acc))

    def grad(self, x0, u):
        acc = np.diff(u, axis=0) / self.dt
        acc0 = np.pad(acc, ((1, 0), (0, 0)), "constant")
        acc1 = np.pad(acc, ((0, 1), (0, 0)), "constant")
        return 2.0 / self.dt * (acc0 - acc1).flatten() / float(len(acc))

    def hessian(self, x0, u):
        T, du = u.shape
        s = (T - 1) * du

        # main diagonal
        main_diag = np.ones((s,), dtype=float) * 4.0
        main_diag_end = np.ones((du,), dtype=float) * 2.0
        main_diag = np.hstack((main_diag_end, main_diag, main_diag_end)) / self.dt**2 / (T - 1)

        # off diagonal
        off_diag = np.ones((s,), dtype=float) * (-2.0) / self.dt**2 / (T - 1)

        # hessian
        return np.diag(main_diag) + np.diag(off_diag, k=du) + np.diag(off_diag, k=-du)
