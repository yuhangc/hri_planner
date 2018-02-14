#!/usr/bin/env python

import numpy as np


class DynamicsBase(object):
    """
    Base class for dynamics used by IRL, with fixed, discretized time steps
    """
    def __init__(self, dt):
        self.dt = dt
        self.x = None
        self.J = None

    def compute(self, x0, u):
        """
        Compute the trajectory and Jacobian given control actions and initial state
        :param x0: initial state, 1x|X| vector
        :param u: control actions, Tx|U| matrix
        """
        raise Exception("must be implemented by derived classes!")

    def traj(self):
        return self.x

    def jacobian(self):
        return self.J


class ConstAccDynamics(DynamicsBase):
    def __init__(self, dt):
        super(ConstAccDynamics, self).__init__(dt)

        self.J_block = None
        self.T = None
        self.nU = 2
        self.nX = 4

        # linear dynamics matrices
        self.A = np.eye(4) + np.eye(4, 4, 2) * dt
        self.B = np.vstack((np.eye(2) * 0.5 * dt**2, np.eye(2) * dt))

    def compute(self, x0, u):
        """
        Compute the trajectory and Jacobian given control actions and initial state
        :param x0: initial state, state consists of (x1, y1, vx1, vy1, ..., xm, ym, vxm, vym)
        :param u: control actions, acceleration in this case, consists of (ax1, ay1, ..., axm, aym)xT
        """
        self.T = u.shape[0]

        # compute trajectory
        self.x = np.zeros((self.T, self.nX))
        self.x[-1] = x0

        for t in range(self.T):
            self.x[t] = np.dot(self.A, self.x[t-1]) + np.dot(self.B, u[t])

        # compute Jacobian
        self.J = np.zeros((self.T * self.nX, self.T * self.nU))

        for t2 in range(self.T):
            for t1 in range(t2, self.T):
                x1 = t1 * self.nX
                y1 = t2 * self.nU

                if t1 == t2:
                    self.J[x1:(x1+self.nX), y1:(y1+self.nU)] = self.B
                else:
                    self.J[x1:(x1+self.nX), y1:(y1+self.nU)] = \
                        np.dot(self.A, self.J[(x1-self.nX):x1, y1:(y1+self.nU)])
