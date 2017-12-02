#!/usr/bin/env python

import numpy as np
from dynamics import LinearDynamics


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
    def f(self, x, u):
        """
        :param x: Tx(|A|x|X|) matrix 
        :param u: Tx(|A|x|U|) matrix
        """
        return -np.sum(np.square(u)) / len(u)

    def grad(self, x, u):
        """ 
        :return: Tx|A|x|U| vector of the gradient with respect to u 
        """
        return -2.0 * u.flatten() / len(u)

    def hessian(self, x, u):
        """ 
        :return: (Tx|A|x|U|)^2 matrix of the Hessian with respect to u 
        """
        return -2.0 * np.eye(u.size, dtype=float) / len(u)


# FIXME: don't use this feature for now
# TODO: need to double check the dimensionality before using
class Acceleration(FeatureBase):
    def __init__(self, dt):
        super(Acceleration, self).__init__()
        self.dt = dt

    def f(self, x, u):
        acc = np.diff(u, axis=0) / self.dt
        return np.sum(np.square(acc)) / float(len(acc))

    def grad(self, x, u):
        acc = np.diff(u, axis=0) / self.dt
        acc0 = np.pad(acc, ((1, 0), (0, 0)), "constant")
        acc1 = np.pad(acc, ((0, 1), (0, 0)), "constant")
        return 2.0 / self.dt * (acc0 - acc1).flatten() / float(len(acc))

    def hessian(self, x, u):
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


class GoalReward(FeatureBase):
    def __init__(self, dyn, x_goal, R):
        """
        Implements an exponetially decaying reward centered at goal position
        :param dyn: Dynamic update function
        :param x_goal: Goals for each agent |A|x|X| matrix
        :param R: Decaying radius
        """
        super(GoalReward, self).__init__()
        self.dyn = dyn
        self.x_goal = x_goal
        self.R2 = R**2

        self.nA, self.nX = x_goal.shape

        # save intermediate calculations
        self.r_matrix = None

    def f(self, x, u):
        self.r_matrix = np.zeros((len(x), self.nA))

        for a in range(self.nA):
            xa = x[:, a*self.nX:a*(self.nX+1)] - self.x_goal[a]
            self.r_matrix[:, a] = np.exp(-np.sum(np.square(xa), axis=1) / self.R2)

        return np.sum(self.r_matrix)

    def grad(self, x, u):
        return np.dot(self.dyn.jacobian().transpose(), self.grad_x(x, u))

    def hessian(self, x, u):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self.hessian_x(x, u), self.dyn.jacobian()))

    def grad_x(self, x, u):
        # make sure that the intermediate calculation is there
        if self.r_matrix is None:
            self.f(x, u)

        # calculate gradient
        grad = np.zeros_like(x)

        for a in range(self.nA):
            xa = x[:, a*self.nX:a*(self.nX+1)] - self.x_goal[a]
            grad[:, a*self.nX:a*(self.nX+1)] = \
                self.r_matrix[:, a] * (2.0 / self.R2 * (self.x_goal[a] - xa))

        return grad.flatten()

    def hessian_x(self, x, u):
        # make sure that the intermediate calculation is there
        if self.r_matrix is None:
            self.f(x, u)

        # calculate Hessian
        hess = np.zeros(x.size())

        for t in range(len(x)):
            for a in range(self.nA):
                hx = t * (self.nA * self.nX) + a * self.nX
                H_ta = hess[hx:hx+self.nX, hx:hx+self.nX]
                x_ta = x[t, a*self.nX:a*(self.nX+1)] - self.x_goal[a]

                H_ta = self.r_matrix[t, a] * 4.0 / self.R2**2 * np.outer(x_ta, x_ta)
                H_ta += -2.0 / self.R2 * self.r_matrix[t, a] * np.eye(self.nX)

        return hess


class CollisionHR(FeatureBase):
    def __init__(self):
        pass
