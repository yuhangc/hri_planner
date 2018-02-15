#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from data_filter import OptimalFilter


def wrap2pi(ang):
    while ang >= np.pi:
        ang -= 2.0 * np.pi
    while ang < -np.pi:
        ang += 2.0 * np.pi
    return ang


class DataLoader(object):
    def __init__(self):
        # data dimensions
        self.dt_raw = 0.0
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
        self.dt = 1.0 / float(self.fps)

        # filter
        self.traj_filter = None
        self.T_block = 19
        self.n_block = 0
        self.t_block = []
        self.xh_block = []
        self.xr_block = []
        self.uh_block = []
        self.ur_block = []
        self.xh0 = []
        self.xr0 = []

    def load_data_raw(self, data_path, demo_id, max_range=-1):
        # load the meta data info
        file_name = data_path + "/meta_data.txt"
        self.meta_data = np.loadtxt(file_name, skiprows=1)

        self.dt_raw = 1.0 / self.meta_data[0]
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

    def select_data(self):
        self.xh = np.zeros((self.T, self.nX))

        # only use position information from human
        for a in range(self.nA):
            stx = a * self.nXs
            stx_raw = a * self.nXf
            self.xh[:, stx:(stx+2)] = self.xh_raw[:, stx_raw:(stx_raw+2)]
            self.xh[:, (stx+2):(stx+4)] = self.xh_raw[:, (stx_raw+3):(stx_raw+5)]

        # use position and orientation of robot
        self.xr = self.xr_raw[:, 0:3].copy()

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

            # plot circles for goal
            ax.add_patch(mpatch.Circle((self.x_goal[0, xs], self.x_goal[0, xs+1]), 0.25, fill=False))
            ax.add_patch(mpatch.Circle((self.x_goal[1, xs], self.x_goal[1, xs+1]), 0.25, fill=False))

        ax.set_xlabel("x(m)")
        ax.set_ylabel("y(m)")
        plt.axis("equal")
        # plt.show()

    def segment_trajectories(self):
        segs = None
        for a in range(self.nA):
            stx = a * self.nXs
            xh = self.xh[:, stx:(stx+2)]
            # calculate distances to goals
            dist_goal0 = np.linalg.norm(xh - np.tile(self.x_goal[0, stx:(stx+2)], (xh.shape[0], 1)), axis=1)
            dist_goal1 = np.linalg.norm(xh - np.tile(self.x_goal[1, stx:(stx+2)], (xh.shape[0], 1)), axis=1)

            # thresholding
            dist_goal0 = dist_goal0 < 1.0
            dist_goal1 = dist_goal1 < 1.0

            # find transitions
            dist_goal0 = np.diff(dist_goal0)
            dist_goal1 = np.diff(dist_goal1)
            pos0 = np.where(dist_goal0 >= 1.0)[0]
            pos1 = np.where(dist_goal1 >= 1.0)[0]

            # find boundaries
            new_seg = []
            for a, b in zip(pos0, pos1):
                new_seg.append(np.array([min(a, b), max(a, b)]))
            new_seg = np.asarray(new_seg)

            # merge all segs
            if segs is None:
                segs = np.copy(new_seg)
            else:
                segs[:, 0] = np.min(np.vstack((segs[:, 0], new_seg[:, 0])), axis=0)
                segs[:, 1] = np.max(np.vstack((segs[:, 1], new_seg[:, 1])), axis=0)

            # print segs
            # print segs[:, 1] - segs[:, 0]
            # print segs.shape

            # fig, axes = plt.subplots()
            # axes.plot(dist_goal0, '-b')
            # axes.plot(dist_goal1, '--b')
            # plt.show()

        # generate all trajectory segments
        for a, b in segs:
            len_block = b - a + 1

            # fill up the block to length T_block
            ext = (self.T_block - len_block) / 2
            a -= ext
            b += self.T_block - len_block - ext

            print a, b, b - a + 1

            # append data
            self.xh_block.append(self.xh[a:(b+1)])
            self.xr_block.append(self.xr[a:(b+1)])
            self.n_block += 1

    def filter_human_trajectories(self, w):
        # create a filter
        self.traj_filter = OptimalFilter(self.dt, w, self.T_block-1)

        self.x_goal = []

        # filter all trajectories
        for i in range(self.n_block):
            traj = self.xh_block[i]

            # set goal
            self.xh0.append(traj[0])
            self.x_goal.append(traj[self.T_block-1, 0:2])
            self.traj_filter.set_end_points(traj[0], traj[self.T_block-1])

            # filter
            x_filtered, u = self.traj_filter.filter_data(traj[1:])
            x_plt = np.asarray(x_filtered)
            u_plt = np.asarray(u)

            # plot for verification
            fig, axes = plt.subplots(3, 1)

            axes[0].plot(x_plt[:, 0], '-k')
            axes[0].plot(x_plt[:, 1], '--k')
            axes[0].plot(traj[1:, 0], '-b')
            axes[0].plot(traj[1:, 1], '--b')

            axes[1].plot(x_plt[:, 2], '-k')
            axes[1].plot(x_plt[:, 3], '--k')
            axes[1].plot(traj[1:, 2], '-b')
            axes[1].plot(traj[1:, 3], '--b')

            axes[2].plot(u_plt[:, 0], '-k')
            axes[2].plot(u_plt[:, 1], '--k')

            fig, axes = plt.subplots()
            axes.plot(x_plt[:, 0], x_plt[:, 1], '-k', lw=2, marker='o', markersize=10, fillstyle="none")
            axes.plot(traj[1:, 0], traj[1:, 1], '-b', lw=2, marker='o', markersize=10, fillstyle="none")

            plt.show()

            # update trajectory
            self.xh_block[i] = np.asarray(x_filtered)
            self.uh_block.append(np.asarray(u))

            print "optimized trajectory: ", i

        # process the goal
        x_goal = np.zeros((2, 2))
        counter = np.zeros((2,))

        for i, goal in enumerate(self.x_goal):
            x_goal[i % 2] += goal
            counter[i % 2] += 1

        # take the average
        x_goal[0] /= counter[0]
        x_goal[1] /= counter[1]

        for i in range(self.n_block):
            self.x_goal[i] = x_goal[i % 2]

    def filter_robot_trajectories(self, offset):
        for i, xr in enumerate(self.xr_block):
            ur = np.zeros((xr.shape[0]-1, self.nUr))
            for t in range(self.T_block):
                # fix the offset between the marker and the robot
                th = xr[t, 2]
                R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
                xr[t, 0:2] += np.dot(R, offset)

                # calculate control
                if t > 0:
                    ur[t-1, 0] = np.linalg.norm(xr[t, 0:2] - xr[t-1, 0:2]) / self.dt
                    ur[t-1, 1] = wrap2pi(xr[t, 2] - xr[t-1, 2]) / self.dt

            self.ur_block.append(ur)
            self.xr0.append(xr[0])
            self.xr_block[i] = xr[1:]

            # axes.plot(xr[:, 0], xr[:, 1], 'k', lw=2)
            #
            # fig, axes = plt.subplots()
            # axes.plot(xr[:, 2], 'b-')
            # axes.plot(ur[:, 0], 'k-')
            # axes.plot(ur[:, 1], 'k--')

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
            if self.nA > 1:
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
            else:
                axes[0].plot(t, x_h[:, stx], '-k', lw=1, label="pos_x")
                axes[1].plot(t, x_h[:, stx+2], '-b', lw=1, label="vel_x")
                axes[2].plot(t, u_h[:, stu], '-r', lw=1, label="acc_x")
                axes[0].plot(t, x_h[:, stx+1], '--k', lw=1, label="pos_y")
                axes[1].plot(t, x_h[:, stx+3], '--b', lw=1, label="vel_y")
                axes[2].plot(t, u_h[:, stu+1], '--r', lw=1, label="acc_y")

                # add title/legends
                axes[0].set_title("positions subject" + str(a))
                axes[1].set_title("velocities subject" + str(a))
                axes[2].set_title("acceleration subject" + str(a))
                axes[0].legend()
                axes[1].legend()
                axes[2].legend()

        plt.show()

    def save_trajectories(self, save_path):
        for i in range(self.n_block):
            # open file
            file_name = save_path + "/block" + str(i) + ".txt"

            data_to_save = np.hstack((self.xh_block[i], self.uh_block[i], self.xr_block[i], self.ur_block[i]))
            np.savetxt(file_name, data_to_save, delimiter=',')

        # save the initial conditions
        self.xh0 = np.asarray(self.xh0)
        self.xr0 = np.asarray(self.xr0)
        np.savetxt(save_path + "/init.txt", np.hstack((self.xh0, self.xr0)), delimiter=',')

        # save the goal
        np.savetxt(save_path + "/goal.txt", np.asarray(self.x_goal), delimiter=',')


if __name__ == "__main__":
    loader = DataLoader()

    # load and plot raw data
    loader.load_data_raw("/home/yuhang/Documents/irl_data/winter18/user0", "_hp", max_range=-1)
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
    # loader.plot_traj_time_stats(loader.t, loader.xh, uh)
    plt.show()

    # segment and filter trajectories
    loader.segment_trajectories()
    loader.filter_robot_trajectories(np.array([0.085, 0]))
    loader.filter_human_trajectories(w=[1.0, 0.1, 0.1, 1.0])

    # save data to file
    loader.save_trajectories("/home/yuhang/Documents/irl_data/winter18/user0/processed/hp")
