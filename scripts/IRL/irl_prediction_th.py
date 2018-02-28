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
        # self.f_cumu.append(features_th.collision_hr(0.3))
        # self.f_cumu.append(features_th.collision_hr_dynamic(0.25, 0.3, 0.5))
        # self.f_cumu.append(features_th.collision_obs(0.3, [2.055939, 3.406737]))
        self.f_cumu.append(features_th.collision_hr(0.05))
        self.f_cumu.append(features_th.collision_hr_dynamic(0.25, 0.3, 0.1))
        self.f_cumu.append(features_th.collision_obs(0.1, [2.055939, 3.406737]))

        # define all the termination features
        self.f_term = []
        self.f_term.append(features_th.goal_reward_term(self.x_goal))

        # define the weights
        self.th_cumu = utils.vector(len(self.f_cumu))
        self.th_term = utils.vector(len(self.f_term))
        # self.th_cumu.set_value(np.array([1.0, 1.0, 0.5, 0.2, 0.5]))
        # self.th_term.set_value(np.array([10.0]))
        self.th_cumu.set_value(np.array([7.0, 20.0, 1.5, 1.0, 1.2]))
        self.th_term.set_value(np.array([45.0]))


class IterativePredictor(IRLPredictor):
    def predict_full(self, uh0, x0, x_goal, xr, ur):
        x0_curr = x0
        t_full = len(xr)

        # set initial control value
        for uh, uh0 in zip(self.uh, uh0):
            uh.set_value(uh0)

        # run iterative plan
        t_curr = 0
        u_opt = []
        while t_curr + self.T < t_full:
            print 'optimizing for time: ', t_curr
            # set initial condition
            self.set_end_points(x0_curr, x_goal)

            # set robot date
            self.set_robot_data(xr[t_curr:(t_curr+self.T)], ur[t_curr:(t_curr+self.T)])

            # plan
            self.optimizer.minimize()

            # record optimal control for current step
            u_opt.append(self.uh[0].get_value())

            # update initial condition
            x0_curr = self.dyn.forward_dyn_np(x0_curr, self.uh[0].get_value(), self.dt)

            # step forward time
            t_curr += 1

        for uh in self.uh[1:]:
            u_opt.append(uh.get_value())

        # recover the full trajectory
        x_opt = []
        x_val = x0
        for uh in u_opt:
            x_val = self.dyn.forward_dyn_np(x_val, uh, self.dt)
            x_opt.append(x_val)

        return np.asarray(x_opt), np.asarray(u_opt)


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
    # predictor.set_end_points(x0, x_goal)
    # predictor.set_robot_data(xr, ur)

    # set initial controls - randomly perturb uh
    u0 = uh + 0.1 * np.random.randn(uh.shape[0], uh.shape[1])

    # plan the trajectory
    # x_opt, u_opt = predictor.predict(u0)
    x_opt, u_opt = predictor.predict_full(u0, x0, x_goal, xr, ur)

    # visualize
    fig, axes = plt.subplots()
    axes.plot(xh[:, 0], xh[:, 1], '--ok', lw=1, fillstyle="none")
    axes.plot(x_opt[:, 0], x_opt[:, 1], '-ok', lw=2, fillstyle="none")
    axes.plot(xr[:, 0], xr[:, 1], '-or', lw=1, fillstyle="none")
    axes.axis("equal")
    plt.show()


def plot_prediction(xh, xr, x_opt, x_goal, obs_data, ax, flag_legend=True, flag_marker=True):
    if flag_marker:
        ax.plot(xh[:, 0], xh[:, 1], '--ok', lw=1, fillstyle="none", label="measured")
        ax.plot(x_opt[:, 0], x_opt[:, 1], '-ok', lw=2, fillstyle="none", label="predicted")
        ax.plot(xr[:, 0], xr[:, 1], '-or', lw=1, fillstyle="none", label="robot")

        ax.plot(x_goal[0], x_goal[1], 'ok', markersize=10)
        ax.plot(xr[-1, 0], xr[-1, 1], 'or', markersize=10)
        ax.plot(obs_data[0], obs_data[1], 'ob', markersize=10)

        thalf = len(x_opt) / 2
        ax.plot(x_opt[thalf, 0], x_opt[thalf, 1], 'ok')
        ax.plot(xr[thalf, 0], xr[thalf, 1], 'or')
    else:
        ax.plot(xh[:, 0], xh[:, 1], '-', color="grey", lw=1, label="measured")
        ax.plot(x_opt[:, 0], x_opt[:, 1], '-k', lw=1, label="predicted")
        ax.plot(xr[:, 0], xr[:, 1], '-r', lw=1, label="robot")

    ax.axis("equal")

    if flag_legend:
        ax.legend()


def predict_and_save_all(predictor, path, save_path, user_list, cond, n_trial):
    for usr in user_list:
        path_full = path + "/user" + str(usr) + "/processed/" + cond

        init_data = np.loadtxt(path_full + "/init.txt", delimiter=",")
        goal_data = np.loadtxt(path_full + "/goal.txt", delimiter=",")
        obs_data = np.loadtxt(path_full + "/obs.txt", delimiter=',')

        for id in range(n_trial):
            x0 = init_data[id, 0:4]
            x_goal = np.zeros_like(x0)
            x_goal[0:2] = goal_data[id]

            file_name = path_full + "/block" + str(id) + ".txt"
            traj = np.loadtxt(file_name, delimiter=",")

            xh = traj[:, 0:4]
            uh = traj[:, 4:6]
            xr = traj[:, 6:9]
            ur = traj[:, 9:11]

            # set initial controls - randomly perturb uh
            u0 = uh + 0.1 * np.random.randn(uh.shape[0], uh.shape[1])

            # plan the trajectory
            x_opt, u_opt = predictor.predict_full(u0, x0, x_goal, xr, ur)

            # plot and save the trajectory
            fig, axes = plt.subplots()
            plot_prediction(xh, xr, x_opt, x_goal, obs_data, axes)

            # plt.show()
            fig.savefig(save_path + "/" + cond + "/user" + str(usr) + "_demo" + str(id) + ".png")


def test_param_robustness(weights, predictor, path, trial):
    # load the data
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")
    goal_data = np.loadtxt(path + "/goal.txt", delimiter=",")
    obs_data = np.loadtxt(path + "/obs.txt", delimiter=',')

    traj = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=",")

    xh = traj[:, 0:4]
    uh = traj[:, 4:6]
    xr = traj[:, 6:9]
    ur = traj[:, 9:11]

    # extract common info
    x0 = init_data[trial, 0:4]
    x_goal = np.zeros_like(x0)
    x_goal[0:2] = goal_data[trial]

    u0 = uh + 0.1 * np.random.randn(uh.shape[0], uh.shape[1])

    # do a nominal prediction first
    n_cumu = 5
    n_term = 1
    predictor.set_param(weights[0:n_cumu], weights[n_cumu:(n_cumu+n_term)])

    x_opt, u_opt = predictor.predict_full(u0, x0, x_goal, xr, ur)

    fig, axes = plt.subplots()
    plot_prediction(xh, xr, x_opt, x_goal, obs_data, axes)
    plt.show()

    # loop through the parameters
    feature_name = ["vel", "acc", "collisionHR1", "collisionHR2", "collisionObs", "goal"]
    n_sample = 7

    lb = [-1, -1, -1, -1, -1, -1]
    ub = [1, 1, 1, 1, 1, 1]

    fig, axes = plt.subplots(len(weights), n_sample)
    for i in range(len(weights)):
        print "testing paremeter: " + feature_name[i]
        for k, wi in enumerate(np.logspace(lb[i], ub[i], n_sample)*weights[i]):
            w_test = weights
            w_test[i] = wi

            print w_test
            predictor.set_param(w_test[0:n_cumu], w_test[n_cumu:(n_cumu+n_term)])

            u0 = uh + 0.1 * np.random.randn(uh.shape[0], uh.shape[1])
            x_opt, u_opt = predictor.predict_full(u0, x0, x_goal, xr, ur)

            plot_prediction(xh, xr, x_opt, x_goal, obs_data, axes[i, k],
                            flag_legend=False, flag_marker=False)
            axes[i, k].set_title("w_" + feature_name[i] + " = " + "{:.3f}".format(wi))

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.show()


def traj_dist_avg(x1, x2):
    x_diff = x1 - x2
    dist = np.linalg.norm(x_diff, axis=1)

    return np.sum(dist) / len(dist)


def cross_validation(d_divide, th, data_path):
    pass


if __name__ == "__main__":
    dyn = dynamics_th.DynamicsConstAacc()
    # predictor = IRLPredictor(dyn, [0.5, 10])
    predictor = IterativePredictor(dyn, [0.5, 10])

    # predict_and_save_all(predictor,
    #                      "/home/yuhang/Documents/irl_data/winter18",
    #                      "/home/yuhang/Documents/irl_data/winter18/figures",
    #                      [0, 1, 2, 3], "rp", 20)

    test_param_robustness(np.array([7.0, 20.0, 1.5, 0.0, 1.2, 45.0]), predictor,
                          "/home/yuhang/Documents/irl_data/winter18/user0/processed/hp", 0)
    test_param_robustness(np.array([7.0, 20.0, 1.5, 1.0, 1.2, 45.0]), predictor,
                          "/home/yuhang/Documents/irl_data/winter18/user0/processed/rp", 0)
