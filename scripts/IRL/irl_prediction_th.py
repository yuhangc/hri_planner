#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import features_th
import dynamics_th
import utils


class IRLPredictorBase(object):
    def __init__(self, dyn, meta_data, sign=1):
        """
        :param dyn: dynamics/transition model
        :param meta_data: [time step, planning horizon]
        :param sign: -1 for maximization, 1 for minimization
        """
        self.f_cumu = self.f_term = None
        self.dyn = dyn
        self.th_cumu = self.th_term = None
        self.sign = sign

        # data dimensions
        self.dt = meta_data[0]
        self.T = meta_data[1]

        self.nX = 4
        self.nU = 2
        self.nXr = 3
        self.nUr = 2

        # initial/goal position
        self.x0 = utils.vector(self.nX)
        self.x_goal = utils.vector(self.nX)

        # robot data
        self.xr = [utils.vector(self.nXr) for t in range(self.T)]
        self.ur = [utils.vector(self.nUr) for t in range(self.T)]

        # generate human trajectory function
        self.uh = [utils.vector(self.nU) for t in range(self.T)]
        self.xh = None

        self.gen_traj()

        # generate features
        self.gen_features()

        # optimizer
        if sign > 0:
            self.optimizer = utils.Minimizer(self.reward(), self.uh, debug=False)
        else:
            self.optimizer = utils.Minimizer(-self.reward(), self.uh)

    def gen_features(self):
        raise Exception("Must be implemented by a derived class!")

    def gen_traj(self):
        # generate the trajectory function
        self.xh = []

        x_next = self.x0
        for t in range(self.T):
            x_next = self.dyn(x_next, self.uh[t], self.dt)
            self.xh.append(x_next)

    def set_end_points(self, x0, x_goal):
        self.x0.set_value(np.asarray(x0))
        self.x_goal.set_value(np.asarray(x_goal))

    def set_robot_data(self, xr, ur):
        for t in range(self.T):
            self.xr[t].set_value(xr[t])
            self.ur[t].set_value(ur[t])

    def set_param(self, th_cumu, th_term):
        self.th_cumu.set_value(np.asarray(th_cumu))
        self.th_term.set_value(np.asarray(th_term))

    def reward(self):
        # cumulative rewards
        r_cumu = []
        for i, feature in enumerate(self.f_cumu):
            r_cumu.append(self.th_cumu[i] *
                          sum([feature(self.xh[t], self.uh[t], self.xr[t], self.ur[t]) for t in range(self.T)]))
        # for feature, w in zip(self.f_cumu, self.th_cumu):
        #     r_cumu.append(w * sum([feature(self.xh[t], self.uh[t], self.xr[t], self.ur[t]) for t in range(self.T)]))

        # termination reward
        r_term = [self.th_term[i] * feature(self.xh[self.T-1], self.uh[self.T-1])
                  for i, feature in enumerate(self.f_term)]

        return sum(r_cumu) + sum(r_term)

    def get_plan(self):
        xout = []
        uout = []

        x_val = self.x0.get_value()
        for t in range(self.T):
            u_val = self.uh[t].get_value()
            x_val = self.dyn.forward_dyn_np(x_val, u_val, self.dt)
            xout.append(x_val)
            uout.append(u_val)

        return np.asarray(xout), np.asarray(uout)

    def predict(self, uh0):
        # set initial value
        for uh, uh0 in zip(self.uh, uh0):
            uh.set_value(uh0)

        # optimize
        self.optimizer.minimize()

        return self.get_plan()


class IRLPredictor(IRLPredictorBase):
    def gen_features(self):
        # define all the cumulative features
        self.f_cumu = []
        self.f_cumu.append(features_th.velocity())
        self.f_cumu.append(features_th.acceleration())
        self.f_cumu.append(features_th.collision_hr(0.5))
        self.f_cumu.append(features_th.collision_hr_dynamic(0.3, 0.25, 1.0))
        self.f_cumu.append(features_th.collision_obs(0.5, [2.055939, 3.406737]))

        # define all the termination features
        self.f_term = []
        self.f_term.append(features_th.goal_reward_term(self.x_goal))

        # define the weights
        self.th_cumu = utils.vector(len(self.f_cumu))
        self.th_term = utils.vector(len(self.f_term))
        self.th_cumu.set_value(np.array([1.0, 1.0, 0.5, 0.05, 0.5]))
        self.th_term.set_value(np.array([10.0]))


def predict_single_trajectory(predictor, path, id):
    # load data
    file_name = path + "/init.txt"
    init_data = np.loadtxt(file_name, delimiter=",")
    x0 = init_data[id, 0:4]

    file_name = path + "/block" + str(id) + ".txt"
    traj = np.loadtxt(file_name, delimiter=",")

    xh = traj[:, 0:4]
    uh = traj[:, 4:6]
    xr = traj[:, 6:9]
    ur = traj[:, 9:11]

    x_goal = xh[-1]

    # set goals and data for planner
    predictor.set_end_points(x0, x_goal)
    predictor.set_robot_data(xr, ur)

    # set initial controls - randomly perturb uh
    u0 = uh + 0.0 * np.random.randn(uh.shape[0], uh.shape[1])

    # plan the trajectory
    x_opt, u_opt = predictor.predict(u0)

    # visualize
    fig, axes = plt.subplots()
    axes.plot(xh[:, 0], xh[:, 1], '--ok', lw=1, fillstyle="none")
    axes.plot(x_opt[:, 0], x_opt[:, 1], '-ok', lw=2, fillstyle="none")
    axes.plot(xr[:, 0], xr[:, 1], '-or', lw=1, fillstyle="none")
    axes.axis("equal")
    plt.show()


def traj_dist_avg(x1, x2):
    x_diff = x1 - x2
    dist = np.linalg.norm(x_diff, axis=1)

    return np.sum(dist) / len(dist)


def cross_validation(d_divide, th, data_path):
    pass


if __name__ == "__main__":
    dyn = dynamics_th.DynamicsConstAacc()
    predictor = IRLPredictor(dyn, [0.5, 18])

    predict_single_trajectory(predictor, "/home/yuhang/Documents/irl_data/winter18/pilot3/processed", 0)
    predict_single_trajectory(predictor, "/home/yuhang/Documents/irl_data/winter18/pilot3/processed", 1)
    predict_single_trajectory(predictor, "/home/yuhang/Documents/irl_data/winter18/pilot3/processed", 2)
    predict_single_trajectory(predictor, "/home/yuhang/Documents/irl_data/winter18/pilot3/processed", 3)
    predict_single_trajectory(predictor, "/home/yuhang/Documents/irl_data/winter18/pilot3/processed", 4)
