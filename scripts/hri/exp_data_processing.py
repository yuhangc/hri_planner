#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from plotting_utils import add_arrow


# far and near trials
far_trials = [6, 9, 17, 16]
near_trials = [2, 3, 11, 18]


def visualize_frame(ax, t, xh, xr, robot_plan, pred_hp, pred_rp, xr_goal, xh_goal):
    # plot previous trajectories
    robot_traj = xr[0:t+1]
    human_traj = xh[0:t+1]

    if xh[t, 0] == 0 and xh[t, 1] == 0:
        ax.set_facecolor((1.0, 0.67, 0.62))

    # don't plot human trajectory if no detection
    # if not np.all(human_traj == 0):
    #     # ignore the leading zeros
    #     human_traj = np.column_stack((np.trim_zeros(human_traj[:, 0]), np.trim_zeros(human_traj[:, 1])))
    #
    #     # segment the trajectory based on zeros
    #     idx_last = 0
    #     idx_curr = 0
    #     pred_segs = []
    #     while idx_curr < len(human_traj):
    #         if human_traj[idx_curr, 0] == 0 and human_traj[idx_curr, 1] == 0:
    #             ax.plot(human_traj[idx_last:(idx_curr-1), 0], human_traj[idx_last:(idx_curr-1), 1], "-o",
    #                     color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")
    #             pred_segs.append(human_traj[(idx_curr-2):idx_curr])
    #
    #             # find next non-zero element
    #             while idx_curr < len(human_traj) and human_traj[idx_curr, 0] == 0 and human_traj[idx_curr, 1] == 0:
    #                 idx_curr += 1
    #             idx_last = idx_curr
    #         else:
    #             idx_curr += 1
    #
    #     if human_traj[idx_curr-1, 0] != 0 or human_traj[idx_curr-1, 1] != 0:
    #         ax.plot(human_traj[idx_last:idx_curr, 0], human_traj[idx_last:idx_curr, 1], "-o",
    #                 color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")
    #
    #     # plot the predicted segments
    #     for seg in pred_segs:
    #         ax.plot(seg[:, 0], seg[:, 1], '-o', color=(0.8, 0.8, 0.2), fillstyle="none", lw=1.5)

    # plot the other trajectories
    ax.plot(human_traj[:, 0], human_traj[:, 1], "-o",
            color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")
    ax.plot(robot_traj[:, 0], robot_traj[:, 1], "-o",
            color=(0.3, 0.3, 0.9), fillstyle="none", lw=1.5, label="robot_traj")
    ax.plot(xr_goal[0], xr_goal[1], 'bo', fillstyle="none", markersize=8)
    ax.plot(xh_goal[0], xh_goal[1], 'ko', fillstyle="none", markersize=8)

    ax.plot(robot_traj[t, 3], robot_traj[t, 4], 'o', color=(1.0, 0.6, 0.6), fillstyle="none")
    ax.plot(human_traj[t, 4], human_traj[t, 5], 'o', color=(0.5, 0.5, 0.5), fillstyle="none")

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

    protocol = np.loadtxt(test_dir + "/protocol.txt", delimiter=',')

    xr_goal = protocol[trial_id, 1:4]
    xh_goal = protocol[trial_id, 4:6]

    # plot the trajectories, plans and predictions
    n_rows = (T - 1) / n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols)

    for t in range(min(T, len(xr))):
        row = t / n_cols
        col = t % n_cols
        visualize_frame(axes[row][col], t, xh, xr, plan_data[t], pred_hp_data[t], pred_rp_data[t], xr_goal, xh_goal)

    fig.tight_layout()

    # plot the velocity profile of human for verification
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(xh[:, 2], '--ks')
    axes[0].plot(xh[:, 3], '--k^')
    vel = np.linalg.norm(xh[:, 2:4], axis=1)
    axes[0].plot(vel, '-k')

    axes[1].plot(xh[:, 6], '--ks')
    axes[1].plot(xh[:, 7], '--k^')
    vel = np.linalg.norm(xh[:, 6:8], axis=1)
    axes[1].plot(vel, '-k')

    # plot the belief and costs
    belief_data = np.loadtxt(test_dir + "/belief_hist" + str(trial_id) + ".txt", delimiter=',')
    cost_data = np.loadtxt(test_dir + "/cost_hist" + str(trial_id) + ".txt", delimiter=',')
    tbelief = np.loadtxt(test_dir + "/tstamp_belief" + str(trial_id) + ".txt", delimiter=',')

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(tbelief, belief_data, '-ks', lw=1.5)

    axes[1].plot(cost_data[:, 0], '-bs', lw=1.5, fillstyle="none", label="cost no communication")
    axes[1].plot(cost_data[:, 1], '--b^', lw=1.5, fillstyle="none", label="cost communication")
    axes[1].set_title("robot intent is " + str(int(protocol[trial_id, 7])))
    axes[1].legend()

    plt.show()


def visualize_trial_video_single(path, cond, trial_id, save_figure=False):
    traj_file = path + "/" + cond + "/block" + str(trial_id) + ".txt"
    traj_data = np.loadtxt(traj_file, delimiter=',')

    xh = traj_data[:, 0:4]
    xr = traj_data[:, 6:9]
    ur = traj_data[:, 9:11]

    t_plot = [3, 7, 11, 21]
    nplots = len(t_plot)
    dt = 0.5

    fig, axes = plt.subplots(1, nplots, figsize=(8, 3))
    for i in range(nplots):
        lh = axes[i].plot(xh[0:(t_plot[i]+1), 0], xh[0:(t_plot[i]+1), 1], 'k-', lw=1, label="human")
        lr = axes[i].plot(xr[0:(t_plot[i]+1), 0], xr[0:(t_plot[i]+1), 1], 'r-', lw=1, label="robot")
        add_arrow(lh[0], position=xh[t_plot[i]-1, 0], size=10)
        add_arrow(lr[0], position=xr[t_plot[i]-1, 0], size=10)
        axes[i].axis("equal")
        axes[i].axis([0, 5, -1, 7])
        axes[i].legend()
        axes[i].set_title("t=" + str(t_plot[i]*dt) + "s")

    fig.tight_layout()

    fig1, axes1 = plt.subplots(figsize=(6, 3))
    vh = np.sqrt(xh[:, 2]**2 + xh[:, 3]**2)
    axes1.plot(vh, label="human")
    axes1.plot(ur[:, 0], label="robot")
    axes1.legend()
    fig1.tight_layout()

    if save_figure:
        fig.savefig(path + "/" + cond + "/block" + str(trial_id) + ".pdf")
        fig1.savefig(path + "/" + cond + "/block" + str(trial_id) + "_vel.pdf")
    else:
        plt.show()

    plt.close(fig)
    plt.close(fig1)


def visualize_trial_video(path, cond, trial_id=-1, ntrials=20):
    if trial_id == -1:
        for trial_id in range(ntrials):
            visualize_trial_video_single(path, cond, trial_id, save_figure=True)
    else:
        visualize_trial_video_single(path, cond, trial_id, save_figure=False)


def visualize_user_video(path, cond, priority, nstart=0, ntrials=20):
    # load the protocol file
    proto = np.loadtxt(path + "/" + cond + "/" + "protocol.txt", delimiter=',')
    pp = proto[1:, 8]

    # plot all trajectories with given priority
    fig, ax = plt.subplots()

    priority = (priority == "rp")
    for trial in range(nstart, ntrials):
        if pp[trial] == priority:
            traj_data = np.loadtxt(path + "/trajectories/" + "/" + cond +
                                   "/block" + str(trial) + ".txt", delimiter=',')

            xh = traj_data[:, 0:4]
            xr = traj_data[:, 6:9]

            ax.plot(xh[:, 0], xh[:, 1], lw=1)
            ax.plot(xr[:, 0], xr[:, 1], lw=1)

    ax.axis("equal")

    plt.show()


def plot_comm_region(path, cond, human_traj_id):
    # load data
    file_name = path + "/data/comm_actions_" + cond + ".txt"
    comm_data = np.loadtxt(file_name, delimiter=',')

    init_data = np.loadtxt(path + "/init.txt", delimiter=',')
    goal_data = np.loadtxt(path + "/goal.txt", delimiter=',')

    # pre-defined human traj
    traj_data = np.loadtxt(path + "/../test" + str(human_traj_id) + ".txt", delimiter=",")

    if cond == "hp":
        acomm = 0
    else:
        acomm = 1

    pts_comm = init_data[np.where(comm_data == acomm)]
    pts_no_comm = init_data[np.where(comm_data != acomm)]
    
    plt.plot(pts_comm[:, 0], pts_comm[:, 1], 'r.')
    plt.plot(pts_no_comm[:, 0], pts_no_comm[:, 1], 'b.')
    plt.plot(goal_data[0], goal_data[1], 'bo', fillstyle="none")
    plt.plot(traj_data[:, 0], traj_data[:, 1], '-', color=(0.5, 0.5, 0.5))

    plt.show()


def visualize_velocities_all(path_root, user_list, cond_list, nstart=10, ntrials=20):
    vr_hp = [[], [], []]
    vr_rp = [[], [], []]
    drel_hp = [[], [], []]
    drel_rp = [[], [], []]

    # load all data
    t_st = 0
    t_int = 16

    for usr in user_list:
        path = path_root + "/user" + str(usr)
        for num, cond in enumerate(cond_list):
            proto = np.loadtxt(path + "/" + cond + "/" + "protocol.txt", delimiter=',')
            pp = proto[1:, 8]

            for trial in range(nstart, ntrials):
                if trial in far_trials or trial in near_trials:
                    continue

                traj_data = np.loadtxt(path + "/trajectories/" + cond +
                                       "/block" + str(trial) + ".txt", delimiter=',')
                vr = traj_data[:, 9]
                xh = traj_data[:, 0:4]
                xr = traj_data[:, 6:9]
                drel = np.linalg.norm(xh[:, 0:2] - xr[:, 0:2], axis=1)

                if pp[trial] == 0:
                    vr_hp[num].append(vr[:t_int])
                    drel_hp[num].append(drel[t_st:t_int])
                elif pp[trial] == 1:
                    vr_rp[num].append(vr[t_st:t_int])
                    drel_rp[num].append(drel[t_st:t_int])

    # plot all
    fig, axes = plt.subplots(3, 2)
    for nc in range(len(cond_list)):
        for trial in range(len(vr_hp[nc])):
            axes[nc][0].plot(vr_hp[nc][trial])
            axes[nc][1].plot(vr_rp[nc][trial])

    fig, axes = plt.subplots(1, 2)
    drel_hp = np.asarray(drel_hp)
    drel_rp = np.asarray(drel_rp)

    for nc in range(len(cond_list)):
        axes[0].plot(np.mean(vr_hp[nc], axis=0), label=cond_list[nc])
        axes[1].plot(np.mean(vr_rp[nc], axis=0), label=cond_list[nc])

    axes[0].legend()
    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    # visualize_trial("/home/yuhang/Documents/hri_log/exp_data/0506-0/test0", 3)
    # visualize_trial_video("/home/yuhang/Videos/hri_planning/user6/trajectories", "haptics", 13)

    # visualize_trial_video("/home/yuhang/Videos/hri_planning/user5/trajectories", "baseline")
    visualize_velocities_all("/home/yuhang/Documents/hri_log/exp_data",
                             [6],
                             ["haptics", "no_haptics", "baseline"])

    # visualize_user_video("/home/yuhang/Documents/hri_log/exp_data/user0", "no_haptics", "hp", nstart=10)

    # plot_comm_region("/home/yuhang/Documents/hri_log/test_data/test_config7", "hp", 0)
