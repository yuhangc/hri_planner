#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import features_const_vel
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


def predict_single_trajectory(d, th, data_path):
    # create the initializer
    initializer = IRLInitializerHRISimple()

    # load the data
    x, u, xr, ur, T = initializer.load_data(data_path)

    # create the features
    x0 = initializer.x0[d]
    u0 = u[d][0]
    x_goal = initializer.x_goal[d]
    features, dyn = initializer.generate_features_termination_cost(u0, x_goal)
    # features, dyn = initializer.generate_features(x0, u0, x_goal)
    # features, dyn = initializer.generate_features(x0, u0, x_goal, goal_reward="linear")

    # create a reward function
    reward = IRLReward(features[0:4], dyn, th[0:4], initializer.meta_data, sign=-1)
    reward.set_data(x0, xr[d], ur[d])

    # set initial guess to be the actual human action
    u0 = u[d] + 0.1 * np.ones_like(u[d])

    # find the optimal trajectory
    res = minimize(reward, u0.flatten(), method="Newton-CG",
                   jac=reward.grad, hess=reward.hessian,
                   options={'xtol': 1e-6, 'disp': True})
    # res = minimize(reward, u0.flatten(), method="BFGS",
    #                jac=reward.grad, options={'disp': True})

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
    ax.plot(xr[d][:, 0], xr[d][:, 1], '-o', linewidth=2, color="salmon", label="Robot")
    ax.plot(x[d][:, 0], x[d][:, 1], '-^b', linewidth=2, label="Measured")
    ax.plot(x_opt[:, 0], x_opt[:, 1], '--ok', linewidth=2, label="Predicted")
    ax.legend(loc='upper right')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.axis("equal")
    plt.show()


def traj_dist_avg(x1, x2):
    x_diff = x1 - x2
    dist = np.linalg.norm(x_diff, axis=1)

    return np.sum(dist) / len(dist)


def cross_validation(d_divide, th, data_path):
    # create the initializer
    initializer = IRLInitializerHRISimple()

    # load the data
    x, u, xr, ur, T = initializer.load_data(data_path)

    # loop through all data
    nD = len(x)
    dist_avg = np.zeros((nD, ))
    reward_pred = np.zeros((nD, ))
    reward_real = np.zeros((nD, ))

    for d in range(nD):
        # create the features
        x0 = x[d][0]
        u0 = u[d][0]
        x_goal = initializer.x_goal[d]
        features, dyn = initializer.generate_features(x0, u0, x_goal)

        # create a reward function
        reward = IRLReward(features[0:4], dyn, th[0:4], initializer.meta_data, sign=-1)
        reward.set_data(x0, xr[d], ur[d])

        # set initial guess to be the actual human action
        u0 = u[d] + 0.1 * np.ones_like(u[d])

        # find the optimal trajectory
        res = minimize(reward, u0.flatten(), method="Newton-CG",
                       jac=reward.grad, hess=reward.hessian,
                       options={'xtol': 1e-6, 'disp': True})

        u_opt = res.x
        u_opt = u_opt.reshape(u[d].shape)

        # generate trajectory with the planned control
        x_opt = dyn.compute(x0, u_opt)

        # compute "distance" to recorded trajectory
        dist = traj_dist_avg(x_opt, x[d])
        dist_avg[d] = dist

        # compute the "reward" of the predicted and real trajectory
        reward_pred[d] = reward(u_opt)
        reward_real[d] = reward(u[d])

    # separate the training and validation data
    dist_avg_train = dist_avg[0:d_divide]
    reward_pred_train = reward_pred[0:d_divide]
    reward_real_train = reward_real[0:d_divide]

    dist_avg_val = dist_avg[d_divide:]
    reward_pred_val = reward_pred[d_divide:]
    reward_real_val = reward_real[d_divide:]

    # display average results
    print "Average trajectory distance: training ", np.mean(dist_avg_train), \
        ",  validation: ", np.mean(dist_avg_val)

    print "Reward (real, pred): training (", np.mean(reward_real_train), ", ", np.mean(reward_pred_train), ")", \
        ",  validation: (", np.mean(reward_real_val), ", ", np.mean(reward_pred_val), ")"

    # save to file
    data_out = np.vstack((dist_avg_train, reward_real_train, reward_pred_train)).transpose()
    np.savetxt("/home/yuhang/Documents/cs333_project/results_training.txt", data_out, delimiter=", ")

    data_out = np.vstack((dist_avg_val, reward_real_val, reward_pred_val)).transpose()
    np.savetxt("/home/yuhang/Documents/cs333_project/results_validation.txt", data_out, delimiter=", ")


if __name__ == "__main__":
    # th = np.array([-7.0, -0.5, 10.0, -0.3])
    # th = np.array([-5.87034223, -3.31171493, -4.03678902, -0.60558638])  # robot priority, linear goal reward
    # th = np.array([-4.77463267, -2.79225923,  9.38634603, -0.48302254])  # robot priority, exp goal reward
    # th = np.array([-3.46008392, -2.67728274,   9.26344071,  -2.61794507e-03])  # human priority, exp goal
    th = np.array([-4.77463267, -0.1,  -50.0, -0.1])

    predict_single_trajectory(3, th, "/home/yuhang/Documents/irl_data/linear_dyn/human_priority")

    # cross_validation(54, th, "/home/yuhang/Documents/irl_data/linear_dyn/human_priority")
