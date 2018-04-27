#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def load_prediction(load_path, usr_id, cond, trial_id, t_start):
    if t_start > 0:
        file_name = load_path + "/user" + str(usr_id) + "/" + cond + \
                    "/demo" + str(trial_id) + "_t" + str(t_start) + ".txt"
    else:
        file_name = load_path + "/user" + str(usr_id) + "/" + cond + \
                    "/demo" + str(trial_id) + ".txt"

    traj = np.loadtxt(file_name, delimiter=',')

    xh = traj[:, 0:4]
    uh = traj[:, 4:6]

    return xh, uh


def load_measurement(load_path, trial_id):
    file_name = load_path + "/block" + str(trial_id) + ".txt"
    traj = np.loadtxt(file_name, delimiter=",")

    xh = traj[:, 0:4]
    uh = traj[:, 4:6]
    xr = traj[:, 6:9]
    ur = traj[:, 9:11]

    return xh, uh, xr, ur


def load_and_plot(load_path, usr_id, cond, trial_id, meta_data=(15, 6, 4)):
    t_total, t_plan, t_inc = meta_data

    # load the experiment data
    path_full = load_path + "/user" + str(usr_id) + "/processed/" + cond

    goal_data = np.loadtxt(path_full + "/goal.txt", delimiter=",")
    obs_data = np.loadtxt(path_full + "/obs.txt", delimiter=',')

    x_goal = goal_data[trial_id]

    xh, uh, xr, ur = load_measurement(path_full, trial_id)

    # load and plot the figures
    n_plots = (t_total - t_plan) / t_inc + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(9, 4))

    k = 0
    for t in range(0, t_total-t_plan, t_inc):
        x_opt, u_opt = load_prediction(load_path + "/prediction", usr_id, cond, trial_id, t)

        # plot the "past" trajectory
        axes[k].plot(xh[:t+1, 0], xh[:t+1, 1], '-', lw=2, color=(0.8, 0.8, 0.8))
        axes[k].plot(xr[:t+1, 0], xr[:t+1, 1], '-', lw=2, color=(1.0, 0.7, 0.7))

        axes[k].plot(xh[t:, 0], xh[t:, 1], '--k', lw=1, fillstyle="none", label="measured")
        axes[k].plot(xr[t:, 0], xr[t:, 1], '-r', lw=1, fillstyle="none", label="robot")
        axes[k].plot(x_opt[:, 0], x_opt[:, 1], '-k', lw=1, fillstyle="none", label="predicted")

        axes[k].plot(x_goal[0], x_goal[1], 'ok', markersize=8, fillstyle="none")
        axes[k].plot(xr[-1, 0], xr[-1, 1], 'or', markersize=8, fillstyle="none")
        axes[k].plot(obs_data[0], obs_data[1], 'ob', markersize=10)

        axes[k].set_title("t = " + str(t / 2))
        axes[k].axis("equal")

        if k == n_plots-1:
            axes[k].legend()

        k += 1

    fig.tight_layout()
    plt.show()


def traj2d_dist_naive(traj1, traj2):
    return np.mean(np.linalg.norm(traj1[:, 0:2] - traj2[:, 0:2], axis=1))


def traj2d_dist_dtw(traj1, traj2):
    pass


def compute_prediction_err(load_path, save_path, user_list, cond, n_trial, dist_func, meta_data=(15, 6, 2)):
    t_total, t_plan, t_inc = meta_data
    errs = []

    for usr in user_list:
        print "processing user ", usr, " ..."
        path_meas = load_path + "/user" + str(usr) + "/processed/" + cond
        path_pred = load_path + "/prediction"

        n_seg = (t_total - t_plan) / t_inc + 1
        err_usr = np.zeros((n_seg, n_trial[usr]))

        for trial in range(n_trial[usr]):
            xh, uh, xr, ur = load_measurement(path_meas, trial)

            k = 0
            for t in range(0, t_total-t_plan, t_inc):
                x_opt, u_opt = load_prediction(path_pred, usr, cond, trial, t)
                err_usr[k, trial] = dist_func(xh[t:t_total, :], x_opt)
                k += 1

        errs.append(err_usr)

        # save to text file
        file_name = save_path + "/" + cond + "/err_user" + str(usr) + ".txt"
        np.savetxt(file_name, err_usr, delimiter=',')

    return errs


def cross_validation(load_path, save_path, user_list, cond, start_trial, meta_data=(15, 6, 2)):
    t_total, t_plan, t_inc = meta_data
    err_dist = None
    err_seg = []

    for usr in user_list:
        # load data
        file_name = load_path + "/" + cond + "/err_user" + str(usr) + ".txt"
        err_usr = np.loadtxt(file_name, delimiter=',')
        err_usr = err_usr[:, start_trial[usr]:]

        if err_dist is None:
            err_dist = err_usr.copy()
        else:
            err_dist = np.hstack((err_dist, err_usr))

        # compute average test error for each segment
        err_seg_usr = np.mean(err_usr, axis=1)
        err_seg.append(err_seg_usr)

    # plots
    # overall distribution
    fig, axes = plt.subplots(5, 1, figsize=(5, 8))

    n_seg = (t_total - t_plan) / t_inc + 1
    for k in range(n_seg):
        axes[k].plot(err_dist[k], np.ones_like(err_dist[k]), '.k', markersize=2)
        axes[k].tick_params(
                axis='y',           # changes apply to the x-axis
                which='both',       # both major and minor ticks are affected
                left='off',       # ticks along the bottom edge are off
                right='off',          # ticks along the top edge are off
                labelleft='off')  # labels along the bottom edge are off
        axes[k].set_ylabel("$T_{start}$ = " + str(k))
    axes[n_seg-1].set_xlabel("average prediction error (m)")
    axes[0].set_title("Prediction error distribution")
    fig.tight_layout()

    # average error w.r.t. start time
    err_seg_avg = np.mean(err_seg, axis=0)

    fig, ax = plt.subplots(figsize=(3.5, 3))
    ax.plot(err_seg_avg, '-s', fillstyle="none")
    ax.set_xlabel("$T_{start}$ (s)")
    ax.set_ylabel("Average prediction error (m)")
    fig.tight_layout()

    plt.show()


def gen_init_conditions(path, xrange, yrange, resolution, offset):
    x = np.arange(xrange[0], xrange[1], resolution[0])
    y = np.arange(yrange[0], yrange[1], resolution[1])

    xx, yy = np.meshgrid(x, y)
    xx = xx.flatten()
    yy = yy.flatten()

    th = offset[2]
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for i in range(len(xx)):
        x_trans = R.dot(np.array([xx[i], yy[i]])) + np.array([offset[0], offset[1]])
        xx[i] = x_trans[0]
        yy[i] = x_trans[1]

    plt.plot(xx, yy, '.')
    plt.xlim(-1, 5)
    plt.ylim(0.8, 6.2)
    plt.show()

    # save the data
    x_init = np.column_stack((xx, yy, 0.4 * np.ones_like(xx)))
    np.savetxt(path + "/init.txt", x_init, delimiter=',', fmt="%1.3f")


if __name__ == "__main__":
    # load_and_plot("/home/yuhang/Documents/irl_data/winter18", 0, "hp", 0)

    gen_init_conditions("/home/yuhang/Documents/hri_log/test_data/test_config4",
                        (-1, 1), (-2, 2), (0.2, 0.2), (0.5, 3.0, 0.4))

    # compute_prediction_err("/home/yuhang/Documents/irl_data/winter18",
    #                        "/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                        [0, 1, 2, 3], "hp",
    #                        [62, 27, 36, 62],
    #                        traj2d_dist_naive)
    #
    # compute_prediction_err("/home/yuhang/Documents/irl_data/winter18",
    #                        "/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                        [0, 1, 2, 3], "rp",
    #                        [62, 27, 61, 62],
    #                        traj2d_dist_naive)

    # cross_validation("/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                  "/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                  [0, 1, 2, 3], "hp",
    #                  [0, 0, 0, 0])
    #                  # [40, 17, 20, 40])
