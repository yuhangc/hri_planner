#!/usr/bin/env python

import numpy as np
import features


class IRLInitializerHRISimple(object):
    """
    Initialize IRL with the simple HRI experiment data
    Using velocity, goal, and collision avoidance features
    """
    def __init__(self):
        # data dimensions
        self.n_demo = 0
        self.nA = 0
        self.nX = 0
        self.nU = 0
        self.nXr = 0
        self.nUr = 0

    def load_data(self, data_path):
        # load the meta data info
        file_name = data_path + "/meta_data.txt"
        meta_data = np.loadtxt(file_name)

        self.n_demo = meta_data[0]
        self.nA = meta_data[1]
        self.nX = meta_data[2]
        self.nU = meta_data[3]
        self.nXr = meta_data[4]
        self.nUr = meta_data[5]

        x = []
        u = []
        xr = []
        ur = []
        T = []

        # load data
        for d in range(self.n_demo):
            file_name = data_path + "/demo" + str(d) + ".txt"
            demo_data = np.loadtxt(file_name)
            col_start = 0

            # human poses
            x.append(demo_data[:, col_start:(self.nA*self.nX)])
            col_start += self.nA * self.nX

            u.append(demo_data[:, col_start:(col_start + self.nA*self.nU)])
            col_start += self.nA * self.nU

            xr.append(demo_data[:, col_start:(col_start + self.nXr)])
            col_start += self.nXr

            ur.append(demo_data[:, col_start:(col_start + self.nUr)])

            T.append(demo_data.shape[0])

        return self.n_demo, x, u, xr, ur, T

    def generate_rewards(self):
        pass

if __name__ == "__main__":
    # create the features
    f_vel = features.Velocity()
