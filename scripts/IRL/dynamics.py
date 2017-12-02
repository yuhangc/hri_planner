#!/usr/bin/env python

import numpy as np


class LinearDynamics(object):
    """
    Linear dynamics (most simplified): x(t) = x(t-1) + dt * u(t)
    Assuming that state and action have the same dimension: |X|==|U|
    """
    def __init__(self, dt):
        self.dt = dt
        self.x = None
        self.J = None

    def compute(self, x0, u):
        """
        Compute the trajectory and Jacobian given control actions and initial state
        :param x0: initial state, |A|x|X| vector
        :param u: control actions, Tx(|A|x|U|) matrix
        """
        T, nU = u.shape
        nX = x0.shape[0]

        # compute trajectory
        self.x = np.zeros((T, nX))
        self.x[-1] = x0

        for t in range(T):
            self.x[t] = self.x[t - 1] + self.dt * u[t]

        # compute jacobian
        self.J = np.zeros((T*nX, T*nU))

        for tx in range(T):
            for ty in range(tx, T):
                self.J[tx*nX:tx*(nX+1), ty*nU:ty*(nU+1)] = self.dt * np.eye(nX, nU)

    def traj(self):
        return self.x

    def jacobian(self):
        return self.J
