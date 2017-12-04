#!/usr/bin/env python

import numpy as np
from dynamics import LinearDynamics
from distance import EuclideanDist


class FeatureBase(object):
    def __init__(self):
        pass

    def __call__(self, *args):
        return self.f(*args)

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
        return -np.sum(np.square(u))

    def grad(self, x, u):
        """ 
        :return: Tx|A|x|U| vector of the gradient with respect to u 
        """
        return -2.0 * u.flatten()

    def hessian(self, x, u):
        """ 
        :return: (Tx|A|x|U|)^2 matrix of the Hessian with respect to u 
        """
        return -2.0 * np.eye(u.size, dtype=float)


# FIXME: don't use this feature for now
# TODO: need to double check the dimensionality before using
class Acceleration(FeatureBase):
    def __init__(self, dt):
        super(Acceleration, self).__init__()
        self.dt = dt

    def f(self, x, u):
        acc = np.diff(u, axis=0) / self.dt
        return np.sum(np.square(acc))

    def grad(self, x, u):
        acc = np.diff(u, axis=0) / self.dt
        acc0 = np.pad(acc, ((1, 0), (0, 0)), "constant")
        acc1 = np.pad(acc, ((0, 1), (0, 0)), "constant")
        return 2.0 / self.dt * (acc0 - acc1).flatten()

    def hessian(self, x, u):
        T, du = u.shape
        s = (T - 1) * du

        # main diagonal
        main_diag = np.ones((s,), dtype=float) * 4.0
        main_diag_end = np.ones((du,), dtype=float) * 2.0
        main_diag = np.hstack((main_diag_end, main_diag, main_diag_end)) / self.dt**2

        # off diagonal
        off_diag = np.ones((s,), dtype=float) * (-2.0) / self.dt**2

        # hessian
        return np.diag(main_diag) + np.diag(off_diag, k=du) + np.diag(off_diag, k=-du)


class GoalReward(FeatureBase):
    def __init__(self, dyn, x_goal, R):
        """
        Implements an exponetially decaying reward centered at goal position
        :param dyn: Dynamic update function (dyn.compute() is assumed to be called already)
        :param x_goal: Goals for each agent |A|x|X| matrix
        :param R: Decaying radius
        """
        super(GoalReward, self).__init__()
        self.dyn = dyn
        self.x_goal = x_goal
        self.R2 = R**2

        self.nA, self.nX = x_goal.shape
        self.T = None

        # save intermediate calculations
        self.r_matrix = None

    def f(self, x, u):
        self.T = x.shape[0]
        self.r_matrix = np.zeros((self.T, self.nA))

        for a in range(self.nA):
            xa = x[:, a*self.nX:(a+1)*self.nX] - self.x_goal[a]
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
            xa = x[:, a*self.nX:(a+1)*self.nX] - self.x_goal[a]
            grad[:, a*self.nX:(a+1)*self.nX] = \
                self.r_matrix[:, a:(a+1)] * (2.0 / self.R2 * (self.x_goal[a] - xa))

        return grad.flatten()

    def hessian_x(self, x, u):
        # make sure that the intermediate calculation is there
        if self.r_matrix is None:
            self.f(x, u)

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


class CollisionHR(FeatureBase):
    def __init__(self, dist_func, dyn):
        super(CollisionHR, self).__init__()
        self.dist_func = dist_func
        self.dyn = dyn

        self.dists = None
        self.grad_d = None
        self.grad_d_x = None

        self.T = None
        self.nA = None
        self.nX = None

    def f(self, x, u, xr):
        self.T, self.nX = xr.shape
        self.nA = x.shape[1] / self.nX

        self.dists = self.dist_func.compute(x, xr)
        return -np.sum(1.0 / self.dists)

    def grad(self, x, u, xr):
        return -np.dot(self.dyn.jacobian().transpose(), self.grad_x(x, u, xr))

    def hessian(self, x, u, xr):
        return -np.dot(self.dyn.jacobian().transpose(),
                       np.dot(self.hessian_x(x, u, xr), self.dyn.jacobian()))

    def grad_dist(self, x, u, xr):
        if self.dists is None:
            self.f(x, u, xr)

        return 1.0 / self.dists**2

    def hessian_dist(self, x, u, xr):
        if self.dists is None:
            self.f(x, u, xr)

        return -2.0 / self.dists**3

    def grad_x(self, x, u, xr):
        grad = np.zeros_like(x)
        self.grad_d = self.grad_dist(x, u, xr)
        self.grad_d_x = self.dist_func.grad()

        for a in range(self.nA):
            grad[:, a*self.nX:(a+1)*self.nX] = \
                self.grad_d[:, a:(a+1)] * self.grad_d_x[:, a*self.nX:(a+1)*self.nX]

        return grad.flatten()

    def hessian_x(self, x, u, xr):
        # make sure that intermediate calculation is there
        if self.grad_d is None:
            self.grad_x(x, u, xr)

        # calculate Hessian
        hess = np.zeros((x.size, x.size))
        hess_d = self.hessian_dist(x, u, xr)
        hess_d_x = self.dist_func.hessian()

        for t in range(self.T):
            for a in range(self.nA):
                hx = t * (self.nA * self.nX) + a * self.nX
                grad_dx_ta = self.grad_d_x[t, a*self.nX:(a+1)*self.nX]
                hess_dx_ta = hess_d_x[t*self.nX:(t+1)*self.nX, a*self.nX:(a+1)*self.nX]

                hess[hx:hx+self.nX, hx:hx+self.nX] = \
                    np.outer(grad_dx_ta, grad_dx_ta) * hess_d[t, a] + \
                    hess_dx_ta * self.grad_d[t, a]

        return hess

# TODO: implement human-human collision avoidance feature

# TODO: implement human-static obstacle collision avoidance feature


# test the features
if __name__ == "__main__":
    # generate a set of motions
    x0_human = np.array([0.0, 0.0])
    x_goal_human = np.array([[0.0, 7.5]])

    x0_robot = np.array([-2.0, 4.0])
    x_goal_robot = np.array([3.0, 4.0])

    dt = 1.0
    t_end = 10.0
    T = int(t_end / dt)

    vel_human = np.linalg.norm(x_goal_human - x0_human) / t_end
    vel_robot = np.linalg.norm(x_goal_robot - x0_robot) / t_end

    u_h = vel_human * np.hstack((np.zeros((T, 1)), np.ones((T, 1))))
    u_r = vel_robot * np.hstack((np.ones((T, 1)), np.zeros((T, 1))))

    dyn = LinearDynamics(dt)
    dyn_r = LinearDynamics(dt)

    dyn.compute(x0_human, u_h)
    dyn_r.compute(x0_robot, u_r)

    xh = dyn.traj()
    xr = dyn_r.traj()

    np.set_printoptions(precision=3)
    np.set_printoptions(linewidth=np.nan)

    # create the features
    # velocity feature
    f_vel = Velocity()
    print "velocity feature: ", f_vel(xh, u_h)
    print "velocity feature gradient: \n", f_vel.grad(xh, u_h)
    print "velocity feature Hessian: \n", f_vel.hessian(xh, u_h)

    # goal reward
    R = np.linalg.norm(x_goal_human - x0_human)
    f_goal = GoalReward(dyn, x_goal_human, R)
    print "goal reward feature: ", f_goal(xh, u_h)
    print "goal reward feature gradient: \n", f_goal.grad(xh, u_h)
    print "goal reward feature Hessian: \n", f_goal.hessian(xh, u_h)

    # collision avoidance
    dist_func = EuclideanDist()
    f_collision = CollisionHR(dist_func, dyn)
    print "collision feature: ", f_collision(xh, u_h, xr)
    print "collision feature gradient: \n", f_collision.grad(xh, u_h, xr)
    print "collision feature Hessian: \n", f_collision.hessian(xh, u_h, xr)
