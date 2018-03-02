#!/usr/bin/env python

import numpy as np
from scipy.linalg import block_diag
from dynamics import ConstAccDynamics


class FeatureBase(object):
    def __init__(self, dyn):
        self.dyn = dyn

    def __call__(self, *args):
        return self.f(*args)

    def f(self, *args):
        raise Exception("method must be implemented by derived classes!")

    def grad(self, *args):
        raise Exception("method must be implemented by derived classes!")

    def hessian(self, *args):
        raise Exception("method must be implemented by derived classes!")


class Velocity(FeatureBase):
    def f(self, x, u, xr, ur):
        """
        :param x: Tx|X| matrix 
        :param u: Tx|U| matrix
        """
        return np.sum(x[:, 2:4]**2)

    def grad(self, x, u, xr, ur):
        """ 
        :return: Tx|U| vector of the gradient with respect to u 
        """
        return np.dot(self.dyn.jacobian().transpose(), self.grad_x(x))

    def hessian(self, x, u, xr, ur):
        """ 
        :return: (Tx|A|x|U|)^2 matrix of the Hessian with respect to u 
        """
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self.hessian_x(x), self.dyn.jacobian()))

    def grad_x(self, x):
        grad = np.zeros_like(x)
        grad[:, 2:4] = 2.0 * x[:, 2:4]

        return grad.flatten()

    def hessian_x(self, x):
        hess_block = np.zeros((self.dyn.nX, self.dyn.nX))
        hess_block[2, 2] = 2.0
        hess_block[3, 3] = 2.0

        blocks = [hess_block] * self.dyn.T

        return block_diag(*blocks)


class Acceleration(FeatureBase):
    def f(self, x, u, xr, ur):
        return np.sum(u**2)

    def grad(self, x, u, xr, ur):
        g = 2.0 * u
        return g.flatten()

    def hessian(self, x, u, xr, ur):
        # hessian
        return 2.0 * np.eye(u.size)


class GaussianReward(FeatureBase):
    def __init__(self, dyn, R):
        super(GaussianReward, self).__init__(dyn)
        self.R2 = R**2

        self.nX = dyn.nX
        self.T = dyn.T

    def _f(self, x, x_target):
        x_diff = x[:, 0:2] - x_target
        return np.sum(np.exp(-np.sum(np.square(x_diff), axis=1) / self.R2))

    def _grad_x(self, x, x_target):
        x_diff = x[:, 0:2] - x_target

        # calculate gradient
        grad = np.zeros_like(x)
        grad[:, 0:2] = np.exp(-np.sum(np.square(x_diff), axis=1) / self.R2).reshape(self.T, 1) * \
                       (-2.0 / self.R2 * x_diff)

        return grad.flatten()

    def _hessian_x(self, x, x_target):
        x_diff = x[:, 0:2] - x_target
        r = np.exp(-np.sum(np.square(x_diff), axis=1) / self.R2)

        # calculate Hessian
        hess = np.zeros((x.size, x.size))

        for t in range(self.T):
            tx = t * self.nX
            hess[tx:(tx+2), tx:(tx+2)] = r[t] * 4.0 / self.R2**2 * np.outer(x_diff[t], x_diff[t])
            hess[tx:(tx+2), tx:(tx+2)] += -2.0 / self.R2 * r[t] * np.eye(2)

        return hess


class InverseDistReward(FeatureBase):
    def __init__(self, dyn, offset):
        super(InverseDistReward, self).__init__(dyn)

        self.offset = offset
        self.nX = dyn.nX
        self.T = dyn.T

    def _f(self, x, x_target):
        x_diff = x[:, 0:2] - x_target
        return np.sum(1.0 / (np.sum(np.square(x_diff), axis=1) + self.offset))

    def _grad_x(self, x, x_target):
        x_diff = x[:, 0:2] - x_target
        d = np.sum(np.square(x_diff), axis=1) + self.offset

        # calculate gradient
        grad = np.zeros_like(x)
        grad[:, 0:2] = -2.0 * x_diff / (d**2).reshape(self.T, 1)

        return grad.flatten()

    def _hessian_x(self, x, x_target):
        x_diff = x[:, 0:2] - x_target
        d = np.sum(np.square(x_diff), axis=1) + self.offset

        # calculate Hessian
        hess = np.zeros((x.size, x.size))

        for t in range(self.T):
            tx = t * self.nX
            hess[tx:(tx+2), tx:(tx+2)] = 8.0 / d[t]**3 * np.outer(x_diff[t], x_diff[t])
            hess[tx:(tx+2), tx:(tx+2)] += -2.0 / d[t]**2 * np.eye(2)

        return hess


class GoalReward(GaussianReward):
    def __init__(self, dyn, R, x_goal):
        """
        Implements an exponetially decaying reward centered at goal position
        :param dyn: Dynamic update function (dyn.compute() is assumed to be called already)
        :param x_goal: goal for human 1x|X| matrix
        :param R: Decaying radius
        """
        super(GoalReward, self).__init__(dyn, R)
        self.x_goal = x_goal

    def f(self, x, u, xr, ur):
        return self._f(x, self.x_goal)

    def grad(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(), self._grad_x(x, self.x_goal))

    def hessian(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self._hessian_x(x, self.x_goal), self.dyn.jacobian()))


class CollisionHRStatic(GaussianReward):
    def __init__(self, dyn, R=0.5):
        super(CollisionHRStatic, self).__init__(dyn, R)

    def f(self, x, u, xr, ur):
        return self._f(x, xr[:, 0:2])

    def grad(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(), self._grad_x(x, xr[:, 0:2]))

    def hessian(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self._hessian_x(x, xr[:, 0:2]), self.dyn.jacobian()))


class CollisionHRDynamic(GaussianReward):
    def __init__(self, dyn, w, l):
        super(CollisionHRDynamic, self).__init__(dyn, 1.0)

        self.w = w
        self.l = l

        self.T = self.dyn.T
        self.dt = 1.0

    def f(self, x, u, xr, ur):
        # transform x and x_target
        x_trans = np.zeros_like(x)
        for t in range(self.T):
            # compute center
            th = xr[t, 2]
            xc = np.array([xr[t, 0] + ur[t, 0] * self.dt * np.cos(th), xr[t, 1] + ur[t, 0] * self.dt * np.sin(th)])

            # compute Gaussian length and width
            gradius = np.array([self.l + ur[t, 0] * 2.0 * self.l, self.w])

            # transform points
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            x_trans[t, 0:2] = np.dot(R, x[t, 0:2] - xc) / gradius

        # use normalized gaussian
        return self._f(x_trans, np.array([0.0, 0.0]))

    def grad(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(), self.grad_x(x, u, xr, ur))

    def hessian(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self.hessian_x(x, u, xr, ur), self.dyn.jacobian()))

    def grad_x(self, x, u, xr, ur):
        # transform x and x_target
        x_trans = np.zeros_like(x)
        J = []
        for t in range(self.T):
            # compute center
            th = xr[t, 2]
            xc = np.array([xr[t, 0] + ur[t, 0] * self.dt * np.cos(th), xr[t, 1] + ur[t, 0] * self.dt * np.sin(th)])

            # compute Gaussian length and width
            gradius = np.array([self.l + ur[t, 0] * 2.0 * self.l, self.w])

            # transform points
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            x_trans[t, 0:2] = np.dot(R, x[t, 0:2] - xc) / gradius

            # store intermediate computation
            R[0] /= gradius[0]
            R[1] /= gradius[1]
            J.append(R)

        # get normalized Gaussian gradient
        grad = self._grad_x(x_trans, np.array([0.0, 0.0]))

        # transform by rotation matrix
        for t in range(self.T):
            stx = t * self.nX
            grad[stx:(stx+2)] = np.dot(J[t].transpose(), grad[stx:(stx+2)])

        return grad

    def hessian_x(self, x, u, xr, ur):
        # transform x and x_target
        x_trans = np.zeros_like(x)
        J = []
        for t in range(self.T):
            # compute center
            th = xr[t, 2]
            xc = np.array([xr[t, 0] + ur[t, 0] * self.dt * np.cos(th), xr[t, 1] + ur[t, 0] * self.dt * np.sin(th)])

            # compute Gaussian length and width
            gradius = np.array([self.l + ur[t, 0] * 2.0 * self.l, self.w])

            # transform points
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
            x_trans[t, 0:2] = np.dot(R, x[t, 0:2] - xc) / gradius

            # store intermediate computation
            R[0] /= gradius[0]
            R[1] /= gradius[1]
            J.append(R)

        # get normalized Gaussian hessian
        hess = self._hessian_x(x_trans, np.array([0.0, 0.0]))

        # transform
        for t in range(self.T):
            stx = t * self.nX
            hess[stx:(stx+2), stx:(stx+2)] = np.dot(J[t].transpose(), np.dot(hess[stx:(stx+2), stx:(stx+2)], J[t]))

        return hess


class CollisionObs(GaussianReward):
    def __init__(self, dyn, x_obs, R=0.5):
        super(CollisionObs, self).__init__(dyn, R)
        self.x_obs = x_obs

    def f(self, x, u, xr, ur):
        return self._f(x, self.x_obs)

    def grad(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(), self._grad_x(x, self.x_obs))

    def hessian(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self._hessian_x(x, self.x_obs), self.dyn.jacobian()))


class TerminationReward(FeatureBase):
    def __init__(self, dyn, x_goal):
        super(TerminationReward, self).__init__(dyn)
        self.x_goal = x_goal

        self.T = self.dyn.T
        self.reg = 1e-2

    def f(self, x, u, xr, ur):
        return np.linalg.norm(x[self.T-1, 0:2] - self.x_goal)

    def grad(self, x, u, xr, ur):
        x_diff = x[self.T-1, 0:2] - self.x_goal
        grad_x = x_diff / (np.linalg.norm(x_diff) + self.reg)

        J = self.dyn.jacobian()
        rs = self.dyn.nX * (self.T-1)

        return np.dot(J[rs:(rs+2)].transpose(), grad_x)

    def hessian(self, x, u, xr, ur):
        x_diff = x[self.T-1, 0:2] - self.x_goal
        d = np.linalg.norm(x_diff) + self.reg

        hess_x = -np.outer(x_diff, x_diff) / d**3 + np.eye(2) / d

        rs = self.dyn.nX * (self.T-1)
        J = self.dyn.jacobian()
        Jt = J[rs:(rs+2)]
        return np.dot(Jt.transpose(), np.dot(hess_x, Jt))


# test the features
if __name__ == "__main__":
    # generate a set of motions
    x0_human = np.array([0.0, 0.0, 0.0, 0.0])
    x_goal_human = np.array([0.0, 7.5, 0.0, 0.0])

    x0_robot = np.array([-2.0, 4.0, 0.0])
    x_goal_robot = np.array([3.0, 4.0, 0.0])

    dt = 0.5
    t_end = 5.0
    T = int(t_end / dt)

    # straight line trajectory
    dx = (x_goal_human - x0_human) / T
    x_human = np.outer(np.arange(1, T+1), dx)

    # compute velocity and accelerations
    acc = np.zeros((T, 2))
    for t in range(T):
        if t == 0:
            x_prev = x0_human
        else:
            x_prev = x_human[t-1]

        # acceleration
        acc[t] = (x_human[t, 0:2] - x_prev[0:2] - x_prev[2:4] * dt) / (0.5 * dt**2)

        # velocity
        x_human[t, 2:4] = x_prev[2:4] + dt * acc[t]

    dxr = (x_goal_robot - x0_robot) / T
    x_robot = np.outer(np.arange(1, T+1), dxr)

    vel_robot = np.linalg.norm(x_goal_robot - x0_robot) / T
    u_r = vel_robot * np.hstack((np.ones((T, 1)), np.zeros((T, 1))))

    # human dynamics
    dyn = ConstAccDynamics(dt)
    dyn.compute(x0_human, acc)

    # check if dynamic computation is correct
    xh = dyn.traj()
    err = np.sum((x_human - xh)**2)
    assert err < 1e-3, "Dynamic computation does not match!"

    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=np.nan)

    # create the features
    # velocity feature
    f_vel = Velocity(dyn)
    print "velocity feature: ", f_vel(xh, acc, x_robot, u_r)
    print "velocity feature gradient: \n", f_vel.grad(xh, acc, x_robot, u_r)
    print "velocity feature Hessian: \n", f_vel.hessian(xh, acc, x_robot, u_r)

    # acceleration feature
    f_acc = Acceleration(dyn)
    print "acceleration feature: ", f_acc(xh, acc, x_robot, u_r)
    print "acceleration feature gradient: \n", f_acc.grad(xh, acc, x_robot, u_r)
    print "acceleration feature Hessian: \n", f_acc.hessian(xh, acc, x_robot, u_r)

    # goal reward
    f_goal = TerminationReward(dyn, x_goal_human[0:2])
    print "goal reward feature: ", f_goal(xh, acc, x_robot, u_r)
    print "goal reward feature gradient: \n", f_goal.grad(xh, acc, x_robot, u_r)
    print "goal reward feature Hessian: \n", f_goal.hessian(xh, acc, x_robot, u_r)

    f_goal_cumu = GoalReward(dyn, 0.3, x_goal_human[0:2])
    print "goal reward cumu feature: ", f_goal_cumu(xh, acc, x_robot, u_r)
    print "goal reward cumu feature gradient: \n", f_goal_cumu.grad(xh, acc, x_robot, u_r)
    print "goal reward cumu feature Hessian: \n", f_goal_cumu.hessian(xh, acc, x_robot, u_r)

    # collision avoidance with robot
    f_collision = CollisionHRStatic(dyn, 0.3)
    print "collision feature: ", f_collision(xh, acc, x_robot, u_r)
    print "collision feature gradient: \n", f_collision.grad(xh, acc, x_robot, u_r)
    print "collision feature Hessian: \n", f_collision.hessian(xh, acc, x_robot, u_r)

    # dynamic collision avoidance with robot
    f_collision_dyn = CollisionHRDynamic(dyn, 0.25, 0.25)
    print "dyn collision feature: ", f_collision_dyn(xh, acc, x_robot, u_r)
    print "dyn collision feature gradient: \n", f_collision_dyn.grad(xh, acc, x_robot, u_r)
    print "dyn collision feature Hessian: \n", f_collision_dyn.hessian(xh, acc, x_robot, u_r)

    # collision avoidance with obstacle
    f_obs = CollisionObs(dyn, 0.3, np.array([1.0, 3.5]))
    print "obstacle feature: ", f_obs(xh, acc, x_robot, u_r)
    print "obstacle feature gradient: \n", f_obs.grad(xh, acc, x_robot, u_r)
    print "obstacle feature Hessian: \n", f_obs.hessian(xh, acc, x_robot, u_r)
