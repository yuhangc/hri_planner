#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class HumanTrajGenerator:
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt

        # set some parameters
        self.v_max = 1.0
        self.a_max = 0.6

        self.k_v = 0.8
        self.k_hr = 0.6
        self.th_hr = 2.5
        self.u_std = [0.1, 0.1]

    def generate_path_ignore_robot(self, x_init, x_goal):
        x = []
        u = []
        x_last = x_init

        for t in range(self.T):
            # compute the desired velocity first
            x_diff = x_goal - x_last[0:2]
            vd = self.k_v * x_diff

            # clip the velocity
            v_dir = np.abs(vd) / np.linalg.norm(vd)
            vd = np.clip(vd, -self.v_max * v_dir, self.v_max * v_dir)

            # compute desired acceleration and clip
            ud = (vd - x_last[2:4]) / self.dt
            u_dir = np.abs(ud) / np.linalg.norm(ud)
            ud = np.clip(ud, -self.a_max * u_dir, self.a_max * u_dir)

            # inject noise into control
            dux = np.random.normal(0.0, self.u_std[0], 1)[0]
            duy = np.random.normal(0.0, self.u_std[1], 1)[0]
            ud += np.array([dux, duy])

            # compute the actual velocity and displacement
            x_new = np.zeros((4, ))
            x_new[0:2] = x_last[0:2] + x_last[2:4] * self.dt + 0.5 * ud * self.dt**2
            x_new[2:4] = x_last[2:4] + ud * self.dt

            # append to list
            x.append(x_new)
            u.append(ud)

            x_last = x_new

        # visualize
        x = np.asarray(x)
        u = np.asarray(u)

        fig, ax = plt.subplots()
        ax.plot(x[:, 0], x[:, 1], "-o", color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")
        ax.plot(x_goal[0], x_goal[1], 'ok')
        ax.axis("equal")

        plt.show()

        return x, u

    def generate_path_avoid_robot(self, x_init, x_goal, x_robot):
        x = []
        u = []
        x_last = x_init

        for t in range(self.T):
            # compute the desired velocity first
            x_diff = x_goal - x_last[0:2]
            vd = self.k_v * x_diff

            # clip the velocity
            v_dir = np.abs(vd) / np.linalg.norm(vd)
            vd = np.clip(vd, -self.v_max * v_dir, self.v_max * v_dir)

            # compute desired acceleration and clip
            ud = (vd - x_last[2:4]) / self.dt

            # add in "force/acc" from avoiding robot
            x_rh = x_last[0:2] - x_robot
            dot = np.dot(-x_rh, x_diff)
            if dot > 0 and np.linalg.norm(x_rh) < self.th_hr:
                f_hr = self.k_hr * x_rh

                # make f_hr perpendicular to ud
                f_hr = np.array([-x_diff[1], x_diff[0]]) / np.linalg.norm(x_diff) * np.linalg.norm(f_hr)
            else:
                f_hr = np.array([0.0, 0.0])

            ud += f_hr

            u_dir = np.abs(ud) / np.linalg.norm(ud)
            ud = np.clip(ud, -self.a_max * u_dir, self.a_max * u_dir)

            # inject noise into control
            dux = np.random.normal(0.0, self.u_std[0], 1)[0]
            duy = np.random.normal(0.0, self.u_std[1], 1)[0]
            ud += np.array([dux, duy])

            # compute the actual velocity and displacement
            x_new = np.zeros((4, ))
            x_new[0:2] = x_last[0:2] + x_last[2:4] * self.dt + 0.5 * ud * self.dt**2
            x_new[2:4] = x_last[2:4] + ud * self.dt

            # append to list
            x.append(x_new)
            u.append(ud)

            x_last = x_new

        # visualize
        x = np.asarray(x)
        u = np.asarray(u)

        fig, ax = plt.subplots()
        ax.plot(x[:, 0], x[:, 1], "-o", color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")
        ax.plot(x_goal[0], x_goal[1], 'ok')
        ax.axis("equal")

        plt.show()

        return x, u


def gen_and_save_trajectories(path, trial=-1, method="ignore_robot"):
    # load init and goal data
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")
    goal_data = np.loadtxt(path + "/goal.txt", delimiter=",")

    # create a generator
    generator = HumanTrajGenerator(16, 0.5)

    # generate a single trajectory
    if trial == -1:
        i = 0
        for x_init, x_goal in zip(init_data, goal_data):
            if method == "ignore_robot":
                x, u = generator.generate_path_ignore_robot(x_init[0:4], x_goal[0:2])
            else:
                x_robot = 0.5 * (x_init[0:2] + x_goal[0:2])
                x, u = generator.generate_path_avoid_robot(x_init[0:4], x_goal[0:2], x_robot)

            # save data to file
            np.savetxt(path + "/test" + str(i) + ".txt", np.hstack((x, u)), delimiter=',')
            i += 1
    else:
        x_init = init_data[trial]
        x_goal = goal_data[trial]
        if method == "ignore_robot":
            x, u = generator.generate_path_ignore_robot(x_init[0:4], x_goal[0:2])
        else:
            x_robot = 0.5 * (x_init[0:2] + x_goal[0:2])
            x, u = generator.generate_path_avoid_robot(x_init[0:4], x_goal[0:2], x_robot)

        # save data to file
        np.savetxt(path + "/test" + str(trial) + ".txt", np.hstack((x, u)), delimiter=',')


if __name__ == "__main__":
    # gen_and_save_trajectories("/home/yuhang/Documents/hri_log/test_data")
    gen_and_save_trajectories("/home/yuhang/Documents/hri_log/test_data", trial=4, method="avoid_robot")
