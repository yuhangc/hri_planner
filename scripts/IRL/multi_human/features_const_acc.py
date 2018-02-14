#!/usr/bin/env python

import numpy as np
from scipy.linalg import block_diag
from dynamics import ConstAccDynamics
from distance import EuclideanDist


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
        :param x: Tx(|A|x|X|) matrix 
        :param u: Tx(|A|x|U|) matrix
        """
        vel_sum = 0.0

        for a in range(self.dyn.nA):
            xs = a * self.dyn.nXs + 2
            vel_sum += np.sum(x[:, xs:(xs+2)]**2)

        return vel_sum

    def grad(self, x, u, xr, ur):
        """ 
        :return: Tx|A|x|U| vector of the gradient with respect to u 
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

        for a in range(self.dyn.nA):
            xs = a * self.dyn.nXs + 2
            grad[:, xs:(xs+2)] = 2.0 * x[:, xs:(xs+2)]

        return grad.flatten()

    def hessian_x(self, x):
        hess_block = np.zeros((self.dyn.nXs, self.dyn.nXs))
        hess_block[2, 2] = 2.0
        hess_block[3, 3] = 2.0

        blocks = [hess_block] * self.dyn.T * self.dyn.nA

        return block_diag(*blocks)


class Acceleration(FeatureBase):
    def f(self, x, u, xr, ur):
        return np.sum(u**2)

    def grad(self, x, u, xr, ur):
        g = 2.0 * u
        return g.flatten()

    def hessian(self, x, u, xr, ur):
        # hessian
        return 2.0 * np.eye(self.dyn.T * self.dyn.nU)


class GoalReward(FeatureBase):
    def __init__(self, dyn, x_goal, R):
        """
        Implements an exponetially decaying reward centered at goal position
        :param dyn: Dynamic update function (dyn.compute() is assumed to be called already)
        :param x_goal: Goals for each agent |A|x|X| matrix
        :param R: Decaying radius
        """
        super(GoalReward, self).__init__(dyn)
        self.x_goal = x_goal
        self.R2 = R**2

        self.nA, self.nX = x_goal.shape
        self.T = None

        # save intermediate calculations
        self.r_matrix = None

    def f(self, x, u, xr, ur):
        self.T = x.shape[0]
        self.r_matrix = np.zeros((self.T, self.nA))

        for a in range(self.nA):
            xa = x[:, a*self.nX:(a+1)*self.nX] - self.x_goal[a]
            self.r_matrix[:, a] = np.exp(-np.sum(np.square(xa), axis=1) / self.R2)

        return np.sum(self.r_matrix)

    def grad(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(), self.grad_x(x, u, xr, ur))

    def hessian(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self.hessian_x(x, u, xr, ur), self.dyn.jacobian()))

    def grad_x(self, x, u, xr, ur):
        # make sure that the intermediate calculation is there
        self.f(x, u, xr, ur)

        # calculate gradient
        grad = np.zeros_like(x)

        for a in range(self.nA):
            xa = self.x_goal[a] - x[:, a*self.nX:(a+1)*self.nX]
            # tmp0 = self.x_goal[a] - xa
            # tmp = 2.0 / self.R2 * (self.x_goal[a] - xa)
            grad[:, a*self.nX:(a+1)*self.nX] = \
                self.r_matrix[:, a:(a+1)] * (2.0 / self.R2 * xa)

        return grad.flatten()

    def hessian_x(self, x, u, xr, ur):
        # make sure that the intermediate calculation is there
        self.f(x, u, xr, ur)

        # calculate Hessian
        hess = np.zeros((x.size, x.size))

        for t in range(len(x)):
            for a in range(self.nA):
                hx = t * (self.nA * self.nX) + a * self.nX
                x_ta = x[t, a*self.nX:(a+1)*self.nX] - self.x_goal[a]

                hess[hx:hx+self.nX, hx:hx+self.nX] = \
                    self.r_matrix[t, a] * 4.0 / self.R2**2 * np.outer(x_ta, x_ta)
                hess[hx:hx+self.nX, hx:hx+self.nX] += \
                    -2.0 / self.R2 * self.r_matrix[t, a] * np.eye(self.nX)

        return hess


class TerminationReward(FeatureBase):
    def __init__(self, dyn, x_goal):
        """
        Implements an exponetially decaying reward centered at goal position
        :param dyn: Dynamic update function (dyn.compute() is assumed to be called already)
        :param x_goal: Goals for each agent |A|x|X| matrix
        :param R: Decaying radius
        """
        super(TerminationReward, self).__init__(dyn)
        self.x_goal = x_goal

    def f(self, x, u, xr, ur):
        T = x.shape[0]
        goal_dist = 0.0

        for a in range(self.dyn.nA):
            xs = a * self.dyn.nXs
            goal_dist += np.sum((x[T-1, xs:(xs+self.dyn.nXs)] - self.x_goal[a])**2)

        return goal_dist

    def grad(self, x, u, xr, ur):
        T = x.shape[0]
        grad_x = 2.0 * (x[T-1] - self.x_goal.flatten())

        J = self.dyn.jacobian()
        rs = self.dyn.nX * (T-1)

        return np.dot(J[rs:(rs+self.dyn.nX)].transpose(), grad_x)

    def hessian(self, x, u, xr, ur):
        T = x.shape[0]
        hess_x = 2.0 * np.eye(self.dyn.nX)

        rs = self.dyn.nX * (T-1)
        J = self.dyn.jacobian()
        Jt = J[rs:(rs+self.dyn.nX)]
        return np.dot(Jt.transpose(), np.dot(hess_x, Jt))


class CollisionHR(FeatureBase):
    def __init__(self, dyn, R):
        super(CollisionHR, self).__init__(dyn)

        self.R2 = R**2

    def exp_dist(self, x, x_r):
        x_diff = np.zeros_like(x)
        dists = np.zeros((self.dyn.T, self.dyn.nA))

        for a in range(self.dyn.nA):
            xs = a * self.dyn.nX
            x_diff[:, xs:(xs+2)] = x[:, xs:(xs+2)] - x_r
            dists[:, a] = np.sum(x_diff[:, xs:(xs+2)]**2, axis=1)

        return np.exp(-dists / self.R2), x_diff

    def f(self, x, u, xr, ur):
        dists, x_diff = self.exp_dist(x, xr)
        return np.sum(dists)

    def grad(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(), self.grad_x(x, u, xr, ur))

    def hessian(self, x, u, xr, ur):
        return np.dot(self.dyn.jacobian().transpose(),
                      np.dot(self.hessian_x(x, u, xr, ur), self.dyn.jacobian()))

    def grad_x(self, x, u, xr, ur):
        # useful calculations
        dists, x_diff = self.exp_dist(x, xr)

        grad = np.zeros_like(x)

        for a in range(self.dyn.nA):
            xs = a * self.dyn.nXs
            grad[:, xs:(xs+2)] = -2.0 / self.R2 * x_diff[:, xs:(xs+2)] * dists[:, a:(a+1)]

        return grad.flatten()

    def hessian_x(self, x, u, xr, ur):
        # useful calculations
        dists, x_diff = self.exp_dist(x, xr)

        # calculate Hessian
        hess = np.zeros((x.size, x.size))

        for t in range(self.dyn.T):
            for a in range(self.dyn.nA):
                # useful calculation
                xs = a * self.dyn.nXs
                mul_vec = x[t, xs:(xs+2)] * (-2.0 / self.R2)

                hx = t * self.dyn.nX + a * self.dyn.nXs

                hess[hx:(hx+2), hx:(hx+2)] = np.outer(mul_vec, mul_vec) * dists[t, a] + \
                                             (-2.0 / self.R2 * dists[t, a]) * np.eye(2)

        return hess

# TODO: implement human-human collision avoidance feature

# TODO: implement human-static obstacle collision avoidance feature


# test the features
if __name__ == "__main__":
    # generate a set of motions
    x0_human = np.array([0.0, 0.0, 0.0, 0.0])
    x_goal_human = np.array([[0.0, 7.5, 0.0, 0.0]])

    x0_robot = np.array([-2.0, 4.0])
    x_goal_robot = np.array([3.0, 4.0])

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
    f_goal = TerminationReward(dyn, x_goal_human.reshape(1, x_goal_human.size))
    print "goal reward feature: ", f_goal(xh, acc, x_robot, u_r)
    print "goal reward feature gradient: \n", f_goal.grad(xh, acc, x_robot, u_r)
    print "goal reward feature Hessian: \n", f_goal.hessian(xh, acc, x_robot, u_r)

    # collision avoidance
    f_collision = CollisionHR(dyn, 0.3)
    print "collision feature: ", f_collision(xh, acc, x_robot, u_r)
    print "collision feature gradient: \n", f_collision.grad(xh, acc, x_robot, u_r)
    print "collision feature Hessian: \n", f_collision.hessian(xh, acc, x_robot, u_r)
