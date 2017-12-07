#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import features
import dynamics
import distance
from irl import MaxEntIRLLinReward
from irl import MaxEntIRL


class IRLInitializerHRISimple(object):
    """
    Initialize IRL with the simple HRI experiment data
    Using velocity, goal, and collision avoidance features
    """
    def __init__(self):
        # data dimensions
        self.dt = 0.0
        self.n_demo = 0
        self.nA = 0
        self.nX = 0
        self.nU = 0
        self.nXr = 0
        self.nUr = 0

        # human goals for each demonstration
        self.x_goal = []
        self.x0 = []
        self.u = []

        # meta data
        self.meta_data = None

    def load_data(self, data_path):
        # load the meta data info
        file_name = data_path + "/meta_data.txt"
        self.meta_data = np.loadtxt(file_name, skiprows=1)

        self.dt = self.meta_data[0]
        # self.n_demo = int(self.meta_data[1])
        self.n_demo = 3
        self.nA = int(self.meta_data[2])
        self.nX = int(self.meta_data[3])
        self.nU = int(self.meta_data[4])
        self.nXr = int(self.meta_data[5])
        self.nUr = int(self.meta_data[6])

        # load the goals for each trial
        file_name = data_path + "/human_goal.txt"
        self.x_goal = np.loadtxt(file_name, delimiter=",")

        # load data
        x = []
        self.u = []
        xr = []
        ur = []
        T = []

        for d in range(self.n_demo):
            file_name = data_path + "/demo" + str(d+1) + ".txt"
            demo_data = np.loadtxt(file_name, delimiter=",")
            col_start = 0

            # human poses and vels
            x.append(demo_data[:, col_start:(self.nA*self.nX)])
            col_start += self.nA * self.nX

            self.u.append(demo_data[:, col_start:(col_start + self.nA*self.nU)])
            col_start += self.nA * self.nU

            # robot pose and vel
            xr.append(demo_data[:, col_start:(col_start + self.nX)])
            col_start += self.nXr

            ur.append(demo_data[:, col_start:(col_start + self.nUr)])

            T.append(demo_data.shape[0])

        # save the initial conditions
        for d in range(self.n_demo):
            self.x0.append(x[d][0])

        return x, self.u, xr, ur, T

    def generate_rewards(self):
        """
        :return: a list of reward functions
        """
        r_list = []

        for d in range(self.n_demo):
            f_list = []

            # velocity feature
            f_vel = features.Velocity()
            f_list.append(f_vel)

            # linear dynamics
            dyn = dynamics.LinearDynamics(self.dt)
            dyn.compute(self.x0[d], self.u[d])

            # goal feature
            R = 0.0
            x_diff = self.x_goal[d] - self.x0[d]
            for a in range(self.nA):
                R += np.linalg.norm(x_diff[a*self.nX:(a+1)*self.nX])
            R /= self.nA * 2.0

            f_goal = features.GoalReward(dyn, self.x_goal[d].reshape((self.nA, self.nX)), R)
            f_list.append(f_goal)

            # collision avoidance with robot
            dist_func = distance.EuclideanDist()
            f_collision_hr = features.CollisionHR(dist_func, dyn)
            f_list.append(f_collision_hr)

            # reward function
            rfunc = MaxEntIRLLinReward(f_list)
            r_list.append(rfunc)

        return r_list

    def generate_features(self, x0, x_goal):
        f_list = []

        # velocity feature
        f_vel = features.Velocity()
        f_list.append(f_vel)

        # linear dynamics
        dyn = dynamics.LinearDynamics(self.dt)

        # goal feature
        R = 0.0
        x_diff = x_goal - x0
        for a in range(self.nA):
            R += np.linalg.norm(x_diff[a*self.nX:(a+1)*self.nX])
        R /= self.nA * 2.0

        f_goal = features.GoalReward(dyn, x_goal.reshape((self.nA, self.nX)), R)
        f_list.append(f_goal)

        # collision avoidance with robot
        dist_func = distance.EuclideanDist()
        f_collision_hr = features.CollisionHR(dist_func, dyn)
        f_list.append(f_collision_hr)

        return f_list, dyn


if __name__ == "__main__":
    # create the initializer
    initializer = IRLInitializerHRISimple()

    # create the IRL solver
    irl_solver = MaxEntIRL(initializer)

    # load data and initialize
    irl_solver.init("/home/yuhang/Documents/irl_data/linear_dyn/human_priority")

    # solve the IRL
    th0 = np.array([-5.0, 10.0, -0.1])
    th_opt, lhist = irl_solver.optimize(th0, n_iter=2000, lrate=0.001, verbose=True)

    print "learned parameters: ", th_opt

    # visualize the reward history
    fig, ax = plt.subplots()
    ax.plot(lhist, '-b', linewidth=2)
    plt.show()
