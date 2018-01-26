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
        :param x0: initial state, |A|x|X| vector
        :param u: control actions, Tx(|A|x|U|) matrix
        """
        raise Exception("must be implemented by derived classes!")

    def traj(self):
        return self.x

    def jacobian(self):
        return self.J


class LinearDynamics(DynamicsBase):
    """
    Linear dynamics (most simplified): x(t) = x(t-1) + dt * u(t)
    Assuming that state and action have the same dimension: |X|==|U|
    """
    def __init__(self, dt):
        super(LinearDynamics, self).__init__(dt)

    def compute(self, x0, u, flag_block_jacobian=True):
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
            for ty in range(0, tx+1):
                self.J[tx*nX:(tx+1)*nX, ty*nU:(ty+1)*nU] = self.dt * np.eye(nX, nU)

        return self.x


class ConstAccDynamics(DynamicsBase):
    def __init__(self, dt):
        super(ConstAccDynamics, self).__init__(dt)

        self.J_block = None
        self.T = None
        self.nU = None
        self.nA = None

        # for each agent, state is (x, y, vx, vy)
        self.nXs = 4

        # linear dynamics matrices
        self.A = np.eye(4) + np.eye(4, 4, 2) * dt
        self.B = np.vstack((np.eye(2) * 0.5 * dt**2, np.eye(2) * dt))

    def compute(self, x0, u):
        """
        Compute the trajectory and Jacobian given control actions and initial state
        :param x0: initial state, state consists of (x1, y1, vx1, vy1, ..., xm, ym, vxm, vym)
        :param u: control actions, acceleration in this case, consists of (ax1, ay1, ..., axm, aym)xT
        """
        self.T, self.nU = u.shape
        self.nA = x0.shape[0] / self.nXs

        # compute trajectory
        self.x = np.zeros((self.T, x0.shape[0]))
        self.x[-1] = x0

        for t in range(self.T):
            for a in range(self.nA):
                self.x[t, (self.nXs * a):(self.nXs * (a+1))] = \
                     np.dot(self.A, self.x[t-1, (self.nXs * a):(self.nXs * (a+1))]) + \
                     np.dot(self.B, u[t, (self.nXs * a):(self.nXs * (a+1))])

        # compute Jacobian (blocks)
        