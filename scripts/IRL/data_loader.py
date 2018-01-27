#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class DataLoader(object):
    def __init__(self):
        # data dimensions
        self.dt = 0.0
        self.nA = 0
        self.nX = 0
        self.nU = 0
        self.nXs = 4
        self.nUs = 2
        self.nXr = 3
        self.nUr = 2
        self.nXf = 0
        self.nXrf = 0

        # store the data in variables
        self.t = None
        self.xh_raw = None
        self.xr_raw = None
        self.x_goal = None

        # meta data
        self.meta_data = None

    def load_data_raw(self, data_path, demo_id):
        # load the meta data info
        file_name = data_path + "/meta_data.txt"
        self.meta_data = np.loadtxt(file_name, skiprows=1)

        self.dt = 1.0 / self.meta_data[0]
        self.nA = int(self.meta_data[1])
        self.nXf = int(self.meta_data[2])
        self.nX = self.nA * self.nXs
        self.nU = self.nA * self.nUs
        self.nXrf = int(self.meta_data[3])
        self.nUr = int(self.meta_data[4])

        # load the goals for each trial
        file_name = data_path + "/human_goal.txt"
        self.x_goal = np.loadtxt(file_name, delimiter=",")

        # load data
        file_name = data_path + "/trajectories" + str(demo_id) + ".txt"
        data_raw = np.loadtxt(file_name, delimiter=",")

        # separate human data and robot data
        self.t = data_raw[:, 0]
        self.xh_raw = data_raw[:, 1:(1+self.nA*self.nXf)]
        self.xr_raw = data_raw[:, (1+self.nA*self.nXf):(1+self.nA*self.nXf+self.nXrf)]

    def plot_raw(self):
        # plot the raw human and robot trajectories
        fig, ax = plt.subplots()

        for a in range(self.nA):
            xs = a * self.nXf
            ax.plot(self.xh_raw[:, xs], self.xh_raw[:, xs+1], color=plt.cm.viridis(a*0.5+0.3), lw=2)

        ax.plot(self.xr_raw[:, 0], self.xr_raw[:, 1], '-k', lw=2)
        plt.show()


if __name__ == "__main__":
    loader = DataLoader()

    loader.load_data_raw("/home/yuhang/Documents/irl_data/winter18/human_first", 1)
    loader.plot_raw()
