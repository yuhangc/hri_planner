#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class DataLoader(object):
    def __init__(self):
        # data dimensions
        self.dt = 0.0
        self.T = 0
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
        self.xh = None
        self.xr = None
        self.uh = None
        self.ur = None
        self.t_raw = None
        self.xh_raw = None
        self.xr_raw = None
        self.x_goal = None

        # meta data
        self.meta_data = None
        self.fps_raw = 0
        self.fps = 2

    def load_data_raw(self, data_path, demo_id, max_range=-1):
        # load the meta data info
        file_name = data_path + "/meta_data.txt"
        self.meta_data = np.loadtxt(file_name, skiprows=1)

        self.dt = 1.0 / self.meta_data[0]
        self.fps_raw = int(self.meta_data[0])
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

        # cut data if max_range is set
        if max_range > 0:
            data_raw = data_raw[0:max_range]

        # separate human data and robot data
        self.T = len(data_raw)
        self.t_raw = data_raw[:, 0]
        self.xh_raw = data_raw[:, 1:(1+self.nA*self.nXf)]
        self.xr_raw = data_raw[:, (1+self.nA*self.nXf):(1+self.nA*self.nXf+self.nXrf)]

    def plot_raw(self):
        # plot the raw human and robot trajectories
        fig, ax = plt.subplots()

        for a in range(self.nA):
            xs = a * self.nXf
            ax.plot(self.xh_raw[:, xs], self.xh_raw[:, xs+1], color=plt.cm.viridis(a*0.5+0.3), lw=2)

        ax.plot(self.xr_raw[:, 0], self.xr_raw[:, 1], '-k', lw=2)
        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        plt.axis("equal")
        # plt.show()

    def generate_irl_data(self):
        """
        Method to do all necessary pre-processes for IRL
        :return: 
        """
        pass

    def select_data(self):
        self.xh = np.zeros((self.T, self.nX))

        # only use position information from human
        for a in range(self.nA):
            stx = a * self.nXs
            stx_raw = a * self.nXf
            self.xh[:, stx:(stx+2)] = self.xh_raw[:, stx_raw:(stx_raw+2)]
            self.xh[:, (stx+2):(stx+4)] = self.xh_raw[:, (stx_raw+3):(stx_raw+5)]

        # use position and orientation of robot
        self.xr = self.xr_raw.copy()

    @staticmethod
    def __down_sample_trajectory__(traj, freq_orig, freq_goal):
        step = freq_orig / freq_goal
        if step < 1:
            raise Exception("Desired frequency is larger than original!!")

        return traj[::step]

    def down_sample_trajectories(self):
        self.t = self.__down_sample_trajectory__(self.t_raw, self.fps_raw, self.fps)
        self.xh = self.__down_sample_trajectory__(self.xh, self.fps_raw, self.fps)
        self.xr = self.__down_sample_trajectory__(self.xr, self.fps_raw, self.fps)

    def plot_human_traj(self):
        # plot the raw human and robot trajectories
        fig, ax = plt.subplots()

        for a in range(self.nA):
            xs = a * self.nXs
            ax.plot(self.xh[:, xs], self.xh[:, xs+1], color=plt.cm.viridis(a*0.5+0.3),
                    lw=2, marker='o', markersize=10, fillstyle="none")

        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        plt.axis("equal")
        # plt.show()

    def calculate_human_velacc(self, x0, traj, dt):
        """
        :param x0: initial poses and velocities of size (|A|x4)x1
        :param traj: trajectories (with 0 velocities) of size |T|x(|A|x4)
        :param dt: time step
        :return: traj of size |T|x(|A|x4) and control/acc of size |T|x(|A|x2)
        """
        T = len(traj)
        traj_full = np.zeros((T, self.nX))
        acc = np.zeros((T, self.nU))

        for t in range(T):
            if t == 0:
                x_prev = x0
            else:
                x_prev = traj_full[t-1]

            for a in range(self.nA):
                stx = a * self.nXs
                sta = a * self.nUs

                # acceleration
                acc[t, sta:(sta+self.nUs)] = (traj[t, stx:(stx+self.nUs)] - x_prev[stx:(stx+self.nUs)] -
                                              x_prev[(stx+self.nUs):(stx+self.nXs)] * dt) / (0.5 * dt**2)

                # velocity
                traj_full[t, (stx+self.nUs):(stx+self.nXs)] = x_prev[(stx+self.nUs):(stx+self.nXs)] + \
                                                              dt * acc[t, sta:(sta+self.nUs)]

                # position
                traj_full[t, stx:(stx+self.nUs)] = traj[t, stx:(stx+self.nUs)]

        return traj_full, acc

    def plot_traj_time_stats(self, t, x_h, u_h):
        # create |A| subplots
        fig, axes = plt.subplots(3, self.nA)

        for a in range(self.nA):
            # compute velocity and acceleration magnitude
            stx = a * self.nXs
            stu = a * self.nUs

            # plot the velocity and acceleration
            axes[0, a].plot(t, x_h[:, stx], '-k', lw=1, label="pos_x")
            axes[1, a].plot(t, x_h[:, stx+2], '-b', lw=1, label="vel_x")
            axes[2, a].plot(t, u_h[:, stu], '-r', lw=1, label="acc_x")
            axes[0, a].plot(t, x_h[:, stx+1], '--k', lw=1, label="pos_y")
            axes[1, a].plot(t, x_h[:, stx+3], '--b', lw=1, label="vel_y")
            axes[2, a].plot(t, u_h[:, stu+1], '--r', lw=1, label="acc_y")

            # add title/legends
            axes[0, a].set_title("positions subject" + str(a))
            axes[1, a].set_title("velocities subject" + str(a))
            axes[2, a].set_title("acceleration subject" + str(a))
            axes[0, a].legend()
            axes[1, a].legend()
            axes[2, a].legend()

        plt.show()

    def segment_trajectories(self):
        pass


if __name__ == "__main__":
    loader = DataLoader()

    # load and plot raw data
    loader.load_data_raw("/home/yuhang/Documents/irl_data/winter18/human_first", 1, max_range=5000)
    # loader.plot_raw()

    # select and down sample trajectories
    loader.select_data()
    loader.down_sample_trajectories()
    loader.plot_human_traj()

    # obtain human velocities and accelerations
    # x0 = loader.xh[0]
    # xh_full, uh = loader.calculate_human_velacc(x0, loader.xh, 0.5)
    uh = np.zeros((len(loader.t), loader.nU))

    # plot the time stats of human trajectory
    loader.plot_traj_time_stats(loader.t, loader.xh, uh)
    plt.show()
