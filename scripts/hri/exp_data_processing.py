#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def visualize_frame(ax, t, xh, xr, robot_plan, pred_hp, pred_rp, ):
    # plot previous trajectories
    robot_traj = xr[0:t]
    human_traj = xh[0:t]

    ax.plot(robot_traj[:, 0], robot_traj[:, 1], "-o",
            color=(0.3, 0.3, 0.9), fillstyle="none", lw=1.5, label="robot_traj")
    ax.plot(human_traj[:, 0], human_traj[:, 1], "-o",
            color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")

    # plot the plan
    nxr = 3
    T = len(robot_plan) / nxr
    robot_plan = robot_plan.reshape(T, nxr)
    ax.plot(robot_plan[:, 0], robot_plan[:, 1], "-",
            color=(0.3, 0.3, 0.9, 0.5), lw=1.0, label="robot_plan")

    nxh = 4
    T = len(pred_hp) / nxh
    pred_hp = pred_hp.reshape(T, nxh)
    ax.plot(pred_hp[:, 0], pred_hp[:, 1], "-",
            color=(0.1, 0.1, 0.1, 0.5), lw=1.0, label="human_pred_hp")

    pred_rp = pred_rp.reshape(T, nxh)
    ax.plot(pred_rp[:, 0], pred_rp[:, 1], "--",
            color=(0.1, 0.1, 0.1, 0.5), lw=1.0, label="human_pred_rp")

    # plot the goals
    # ax.plot(self.xr_goal[0], self.xr_goal[1], 'ob')
    # ax.plot(self.xh_goal[0], self.xh_goal[1], 'ok')

    ax.axis("equal")


def visualize_trial(test_dir, trial_id, T=15, n_cols=5):
    # load data
    xr = np.loadtxt(test_dir + "/robot_traj" + str(trial_id) + ".txt", delimiter=',')
    xh = np.loadtxt(test_dir + "/human_traj" + str(trial_id) + ".txt", delimiter=',')

    plan_data = np.loadtxt(test_dir + "/robot_plan" + str(trial_id) + ".txt", delimiter=',')
    pred_hp_data = np.loadtxt(test_dir + "/human_pred_hp" + str(trial_id) + ".txt", delimiter=',')
    pred_rp_data = np.loadtxt(test_dir + "/human_pred_rp" + str(trial_id) + ".txt", delimiter=',')

    # plot the trajectories, plans and predictions
    n_rows = (T - 1) / n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols)

    for t in range(T):
        row = t / n_cols
        col = t % n_cols
        visualize_frame(axes[row][col], t, xh, xr, plan_data[t], pred_hp_data[t], pred_rp_data[t])

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_trial("/home/yuhang/Documents/hri_log/exp_data/test1", 1)
