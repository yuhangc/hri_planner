#!/usr/bin/env python

import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt

import features_th
import dynamics_th
import utils


class MatEntIRLBase(object):
    def __init__(self, dyn, meta_data, n_training):
        self.f_cumu = self.f_term = None
        self.dyn = dyn

        # data dimensions
        self.dt = meta_data[0]
        self.T = meta_data[1]

        self.nX = 4
        self.nU = 2
        self.nXr = 3
        self.nUr = 2

        self.n_training = n_training

        # initial conditions and goals for each demo
        self.x0 = [utils.vector(self.nX) for _ in range(self.n_training)]
        self.x_goal = [utils.vector(self.nX) for _ in range(self.n_training)]

        # controls and robot trajectories
        self.uh = []
        self.xr = []
        self.ur = []

        for i in range(self.n_training):
            self.uh.append([utils.vector(self.nU) for _ in range(self.T)])
            self.xr.append([utils.vector(self.nXr) for _ in range(self.T)])
            self.ur.append([utils.vector(self.nUr) for _ in range(self.T)])

        # generate all human trajectories
        self.xh = []
        for i in range(self.n_training):
            self.xh.append(self.gen_traj(self.x0[i], self.uh[i]))

        # generate features
        self.th = None
        self.gen_features()

        # the likelihood function and optimizer
        self.L = -self.gen_likelihood()     # negate for maximization
        self.optimizer = utils.Minimizer(self.L, [self.th], method='gd', eps=0.1, debug=True, iters=1000, inf_ignore=10)

    def optimize(self, th0):
        self.th.set_value(np.asarray(th0))
        self.optimizer.minimize()

        return self.th.get_value()

    def gen_features(self):
        raise Exception("Must be implemented by a derived class!")

    def load_data(self, path):
        raise Exception("Must be implemented by a derived class!")

    def gen_traj(self, x0, uh):
        # generate the trajectory function
        xh = []

        x_next = x0
        for t in range(self.T):
            x_next = self.dyn(x_next, uh[t], self.dt)
            xh.append(x_next)

        return xh

    def reward(self, id):
        # cumulative rewards
        xh = self.xh[id]
        uh = self.uh[id]
        xr = self.xr[id]
        ur = self.ur[id]

        r_cumu = []
        for i, feature in enumerate(self.f_cumu):
            r_cumu.append(self.th[i] * sum([feature(xh[t], uh[t], xr[t], ur[t]) for t in range(self.T)]))

        # termination reward
        n_cumu = len(r_cumu)
        r_term = [self.th[i+n_cumu] * feature(xh[self.T-1], uh[self.T-1])
                  for i, feature in enumerate(self.f_term)]

        return sum(r_cumu) + sum(r_term)

    def gen_likelihood(self):
        rewards = [self.reward(i) for i in range(self.n_training)]

        L = None
        for i, reward in enumerate(rewards):
            g = utils.grad(reward, self.uh[i])
            H = utils.hessian(reward, self.uh[i])

            I = tt.eye(utils.shape(H)[0])
            reg = utils.vector(1)
            reg.set_value([1e-1])
            H = H - reg[0] * I

            if L is None:
                L = tt.dot(g, tt.dot(tt.nlinalg.MatrixInverse()(H), g))+tt.log(tt.nlinalg.Det()(-H))
            else:
                L = L + tt.dot(g, tt.dot(tt.nlinalg.MatrixInverse()(H), g))+tt.log(tt.nlinalg.Det()(-H))

        return L


class HumanIRL(MatEntIRLBase):
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

        # initialize the weight vector
        self.th = utils.vector(len(self.f_cumu) + len(self.f_term))

    def load_data(self, path):
        # load the initial conditions
        file_name = path + "/init.txt"
        init_data = np.loadtxt(file_name, delimiter=',')

        for i in range(self.n_training):
            # set initial condition
            self.x0[i].set_value(init_data[i, 0:self.nX])

            # load trajectory data
            file_name = path + "/block" + str(i) + ".txt"
            traj_data = np.loadtxt(file_name, delimiter=',')

            for t in range(self.T):
                self.uh[i][t].set_value(traj_data[t, 4:6])
                self.xr[i][t].set_value(traj_data[t, 6:9])
                self.ur[i][t].set_value(traj_data[t, 9:11])


if __name__ == "__main__":
    # create dynamics
    dyn = dynamics_th.DynamicsConstAacc()

    # create an IRL object
    irl_trainer = HumanIRL(dyn, [0.5, 5], 1)
    print "object created..."

    # load data
    irl_trainer.load_data("/home/yuhang/Documents/irl_data/winter18/pilot3/processed")

    # train
    th_opt = irl_trainer.optimize(np.array([1.0, 1.0, 0.5, 0.05, 0.5, 10.0]))
    print th_opt
