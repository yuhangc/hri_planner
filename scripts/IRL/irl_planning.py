#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import features
import dynamics
import distance
from irl_training import IRLInitializerHRISimple


class IRLReward(object):
    def __init__(self, features, dyn, th, meta_data, sign=-1):
        """
        :param sign: -1 for maximization, 1 for minimization
        """
        self.features = features
        self.dyn = dyn
        self.th = th
        self.sign = sign

        self.nfeature = len(features)

        self.x0 = None
        self.xr = None
        self.ur = None

        # data dimensions
        self.dt = meta_data[0]
        self.nA = int(meta_data[2])
        self.nX = int(meta_data[3])
        self.nU = int(meta_data[4])
        self.nXr = int(meta_data[5])
        self.nUr = int(meta_data[6])
        self.T = 0

    def __call__(self, u):
        return self.reward(u)

    def set_data(self, x0, xr, ur):
        self.x0 = x0
        self.xr = xr
        self.ur = ur

        self.T = ur.shape[0]

    def reward(self, u):
        r = 0.0

        # reshape u
        u = u.reshape((self.T, self.nA * self.nX))

        # generate trajectory with control
        x = self.dyn.compute(self.x0, u)

        # sum up all features
        for i in range(self.nfeature):
            r += self.th[i] * self.features[i](x, u, self.xr, self.ur)

        return self.sign * r

    def grad(self, u):
        # reshape u
        u = u.reshape((self.T, self.nA * self.nX))

        # generate trajectory with control
        x = self.dyn.compute(self.x0, u)

        # sum over the gradient of each feature
        grad = np.zeros_like(u).flatten()

        for i in range(self.nfeature):
            grad += self.th[i] * self.features[i].grad(x, u, self.xr, self.ur)

        return self.sign * grad

    def hessian(self, u):
        # reshape u
        u = u.reshape((self.T, self.nA * self.nX))

        # generate trajectory with control
        x = self.dyn.compute(self.x0, u)

        # sum over the hessian of each feature
        hessian = np.zeros((u.size, u.size))

        for i in range(self.nfeature):
            hessian += self.th[i] * self.features[i].hessian(x, u, self.xr, self.ur)

        return self.sign * hessian


if __name__ == "__main__":
    # create the initializer
    initializer = IRLInitializerHRISimple()

    # load the data
    x, u, xr, ur, T = initializer.load_data("/home/yuhang/Documents/irl_data/linear_dyn/human_priority")

    # test and compare against one demo
    d = 1

    # create the features
    x0 = initializer.x0[d]
    x_goal = initializer.x_goal[d]
    features, dyn = initializer.generate_features(x0, x_goal)

    # create a reward function
    th = np.array([-5.0, 10.0, -0.1])
    reward = IRLReward(features[0:2], dyn, th[0:2], initializer.meta_data, sign=-1)
    reward.set_data(x0, xr[d], ur[d])

    # set initial guess to be the actual human action
    u0 = u[d] + 0.1 * np.ones_like(u[d])

    # find the optimal trajectory
    # res = minimize(reward, u0.flatten(), method="Newton-CG",
    #                jac=reward.grad, hess=reward.hessian,
    #                options={'xtol': 1e-8, 'disp': True})
    res = minimize(reward, u0.flatten(), method="BFGS",
                   jac=reward.grad, options={'disp': True})

    u_opt = res.x
    u_opt = u_opt.reshape(u[d].shape)

    # # do simple gradient decent
    # u_opt = u0
    # r_hist = []
    # for iter in range(500):
    #     r = reward(u_opt)
    #     r_hist.append(r)
    #
    #     grad = reward.grad(u_opt)
    #     u_opt -= 0.002 * grad.reshape(u0.shape)
    #
    #     print "reward: ", r, ", gradient mag: ", np.linalg.norm(grad)
    #
    print u_opt
    #
    # fig, ax = plt.subplots()
    # ax.plot(r_hist, '-b', linewidth=2)
    # plt.show()

    # generate trajectory with the planned control
    x_opt = dyn.compute(x0, u_opt)

    # plot the trajectories
    fig, ax = plt.subplots()
    ax.plot(x[d][:, 0], x[d][:, 1], '-b', linewidth=2, label="Measured")
    ax.plot(x_opt[:, 0], x_opt[:, 1], '--ok', linewidth=2, label="Predicted")
    plt.axis("equal")
    plt.show()
