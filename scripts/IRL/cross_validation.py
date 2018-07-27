#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate

from irl_prediction_th import plot_prediction

import sys
sys.path.insert(0, '../hri')

from data_loader import wrap2pi
from plotting_utils import *


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
    fig, axes = plt.subplots(1, n_plots, figsize=(5.5, 2.67))

    k = 0
    for t in range(0, t_total-t_plan, t_inc):
        x_opt, u_opt = load_prediction(load_path + "/prediction", usr_id, cond, trial_id, t)

        # plot the "past" trajectory
        axes[k].plot(xh[:t+1, 0], xh[:t+1, 1], '-', lw=1.5, color=(0.8, 0.8, 0.8))
        axes[k].plot(xr[:t+1, 0], xr[:t+1, 1], '-', lw=1.5, color=(0.8, 0.7, 0.7))

        h_xh = axes[k].plot(xh[t:, 0], xh[t:, 1], '--k', lw=1, fillstyle="none", label="measured")
        h_xopt = axes[k].plot(x_opt[:, 0], x_opt[:, 1], '-k', lw=1, fillstyle="none", label="predicted")
        h_xr = axes[k].plot(xr[t:, 0], xr[t:, 1], '-r', lw=1, fillstyle="none", label="robot")
        add_arrow(h_xh[0], position=xh[-2, 0], size=7)
        add_arrow(h_xr[0], position=xr[-2, 0], size=7)
        add_arrow(h_xopt[0], position=x_opt[-2, 0], size=7)

        # axes[k].plot(x_goal[0], x_goal[1], 'ok', markersize=8, fillstyle="none")
        # axes[k].plot(xr[-1, 0], xr[-1, 1], 'or', markersize=8, fillstyle="none")
        axes[k].plot(obs_data[0], obs_data[1], 'ok', markersize=8)

        axes[k].axis("equal")

        turn_off_axes_labels(axes[k])

        props = dict(boxstyle='square', facecolor='white', edgecolor='w')
        axes[k].text(0.05, 0.04, "t="+str(t*0.5)+"s", transform=axes[k].transAxes, fontsize=14,
                     verticalalignment='bottom', bbox=props)

        k += 1

    axes[0].legend(bbox_to_anchor=(0., -0.2, 2.8, .102), loc=3,
                   ncol=3, mode="expand", borderaxespad=0., fontsize=14, fancybox=False, edgecolor='w')

    fig.subplots_adjust(left=0.03, bottom=0.2, right=0.97, top=0.97, wspace=0.1, hspace=0.2)
    plt.show()


def traj2d_dist_naive(traj1, traj2):
    return np.mean(np.linalg.norm(traj1[:, 0:2] - traj2[:, 0:2], axis=1))


def traj2d_dist_dtw(traj1, traj2):
    pass


def check_wrong_side(traj1, traj2, obs):
    om1 = 0.0
    om2 = 0.0

    for t in range(1, len(traj1)):
        v1 = traj1[t, 0:2] - obs[0:2]
        v2 = traj2[t, 0:2] - obs[0:2]

        v1_last = traj1[t-1, 0:2] - obs[0:2]
        v2_last = traj2[t-1, 0:2] - obs[0:2]

        om1 += wrap2pi(np.arctan2(v1[1], v1[0]) - np.arctan2(v1_last[1], v1_last[0]))
        om2 += wrap2pi(np.arctan2(v2[1], v2[0]) - np.arctan2(v2_last[1], v2_last[0]))

    return om1 * om2 < 0.0


def compute_prediction_err(load_path, save_path, user_list, cond, n_trial, dist_func, meta_data=(15, 6, 2), suff=""):
    t_total, t_plan, t_inc = meta_data
    errs = []

    for usr in user_list:
        print "processing user ", usr, " ..."
        path_meas = load_path + "/user" + str(usr) + "/processed/" + cond
        path_pred = load_path + "/prediction" + suff

        n_seg = (t_total - t_plan) / t_inc + 1
        err_usr = np.zeros((n_seg, n_trial[usr]))
        wrong_side = np.zeros((n_trial[usr], ))

        for trial in range(n_trial[usr]):
            xh, uh, xr, ur = load_measurement(path_meas, trial)

            k = 0
            for t in range(0, t_total-t_plan, t_inc):
                x_opt, u_opt = load_prediction(path_pred, usr, cond, trial, t)
                err_usr[k, trial] = dist_func(xh[t:t_total, :], x_opt[:(t_total-t), :])
                if k == 0:
                    wrong_side[trial] = check_wrong_side(xh[:t_total], x_opt, np.array([2.055939, 3.406737]))
                    # if wrong_side[trial] > 0:
                    #     fig, ax = plt.subplots()
                    #     ax.plot(xh[:, 0], xh[:, 1])
                    #     ax.plot(x_opt[:, 0], x_opt[:, 1])
                    #     plt.show()

                k += 1

        errs.append(err_usr)

        # save to text file
        file_name = save_path + "/" + cond + suff + "/err_user" + str(usr) + ".txt"
        np.savetxt(file_name, err_usr, delimiter=',')
        np.savetxt(save_path + "/" + cond + suff + "/wrong_side" + str(usr) + ".txt", wrong_side, delimiter=', ')

    return errs


def load_err_data(path, user_list, cond, start_trial, flag_remove_outlier=False):
    err_dist = None
    err_cs = None
    err_ws = None

    for usr in user_list:
        # load data
        file_name = path + "/" + cond + "/err_user" + str(usr) + ".txt"
        err_usr = np.loadtxt(file_name, delimiter=',')
        wrong_side = np.loadtxt(path + "/" + cond + "/wrong_side" + str(usr) + ".txt", delimiter=',')
        err_usr = err_usr[:, start_trial[usr]:]
        wrong_side = wrong_side[start_trial[usr]:]

        idx = np.where(wrong_side > 0)
        err_usr_ws = err_usr[:, idx[0]]
        idx = np.where(wrong_side == 0)
        err_usr_cs = err_usr[:, idx[0]]
        if err_dist is None:
            err_dist = err_usr.copy()
            err_cs = err_usr_cs.copy()
            err_ws = err_usr_ws.copy()
        else:
            err_dist = np.hstack((err_dist, err_usr))
            err_cs = np.hstack((err_cs, err_usr_cs))
            err_ws = np.hstack((err_ws, err_usr_ws))

    if flag_remove_outlier:
        idx_outlier = np.where(err_ws > 0.8)
        print idx_outlier
        idx_inlier = np.where(err_ws[2, :] < 0.8)
        err_ws = err_ws[:, idx_inlier[0]]
        err_cs = err_cs[:, idx_inlier[0]]

    return err_dist, err_cs, err_ws


def cross_validation(load_path, save_path, user_list, cond, start_trial, meta_data=(15, 6, 2)):
    t_total, t_plan, t_inc = meta_data

    err_dist, err_cs, err_ws = load_err_data(load_path, user_list, cond, start_trial)

    # plots
    # overall distribution
    fig, axes = plt.subplots(5, 1, figsize=(5, 8))

    n_seg = (t_total - t_plan) / t_inc + 1
    for k in range(n_seg):
        axes[k].plot(err_cs[k], np.ones_like(err_cs[k]), '.k', markersize=2)
        axes[k].plot(err_ws[k], np.ones_like(err_ws[k]), '.r', markersize=2)
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
    err_avg_cs = np.mean(err_cs, axis=1)
    err_std_cs = np.std(err_cs, axis=1)
    err_avg_ws = np.mean(err_ws, axis=1)
    err_std_ws = np.std(err_ws, axis=1)
    x = np.arange(len(err_avg_cs))

    fig, ax = plt.subplots(figsize=(4.5, 3))
    # ax.plot(err_seg_avg, '-s', fillstyle="none")
    ax.errorbar(x, err_avg_cs, yerr=err_std_cs, marker='s', fillstyle="none", ls="-", lw=2,
                color=(0.1, 0.1, 0.6), elinewidth=1.0, capsize=3, capthick=1, label="correct side")
    ax.errorbar(x, err_avg_ws, yerr=err_std_ws, marker='s', fillstyle="none", ls="--", lw=2,
                color=(0.6, 0.1, 0.1), elinewidth=1.0, capsize=3, capthick=1, label="wrong side")
    ax.set_xlabel("$t$ (s)", fontsize=14)
    ax.set_ylabel("Prediction error (m)", fontsize=14)

    ax.legend(frameon=True, fancybox=False, edgecolor='k', fontsize=14)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    fig.tight_layout()
    plt.show()


def cross_validation_all_cond(load_path, user_list, start_trial, meta_data=(15, 6, 2)):
    t_total, t_plan, t_inc = meta_data
    err_dist = []
    err_cs = []
    err_ws = []

    cond_list = ["hp", "rp", "hp_sf", "rp_sf"]
    remove_outlier = [True, True, False, False]
    for cond in cond_list:
        err_disti, err_csi, err_wsi = load_err_data(load_path, user_list, cond, start_trial)
        err_dist.append(err_disti)
        err_cs.append(err_csi)
        err_ws.append(err_wsi)

    # irl and social force model error
    err_cs_irl = np.hstack((err_cs[0], err_cs[1]))
    err_ws_irl = np.hstack((err_ws[0], err_ws[1]))
    err_cs_sf = np.hstack((err_cs[2], err_cs[3]))
    err_ws_sf = np.hstack((err_ws[2], err_ws[3]))

    fig, ax = plt.subplots(figsize=(4.5, 3))

    # average error w.r.t. start time
    err_avg_cs = np.mean(err_cs_irl, axis=1)
    err_std_cs = np.std(err_cs_irl, axis=1)
    err_avg_ws = np.mean(err_ws_irl, axis=1)
    err_std_ws = np.std(err_ws_irl, axis=1)
    x = np.arange(len(err_avg_cs))

    # ax.plot(err_seg_avg, '-s', fillstyle="none")
    ax.errorbar(x, err_avg_cs, yerr=err_std_cs, marker='s', markerfacecolor='w', ls="-", lw=2,
                color=(34/255.0, 144/255.0, 196/255.0), elinewidth=1.0, capsize=3, capthick=1, label="IRL, type I")
    ax.errorbar(x, err_avg_ws, yerr=err_std_ws, marker='o', markerfacecolor='w', ls="--", lw=2,
                color=(34/255.0, 144/255.0, 196/255.0), elinewidth=1.0, capsize=3, capthick=1, label="IRL, type II")

    # average error w.r.t. start time
    err_avg_cs = np.mean(err_cs_sf, axis=1)
    err_std_cs = np.std(err_cs_sf, axis=1)
    err_avg_ws = np.mean(err_ws_sf, axis=1)
    err_std_ws = np.std(err_ws_sf, axis=1)
    x = np.arange(len(err_avg_cs))

    # ax.plot(err_seg_avg, '-s', fillstyle="none")
    ax.errorbar(x, err_avg_cs, yerr=err_std_cs, marker='s', markerfacecolor='w', ls="-", lw=1.5,
                color=(255/255.0, 142/255.0, 47/255.0), elinewidth=1.0, capsize=3, capthick=1, label="SF, type I")
    ax.errorbar(x, err_avg_ws, yerr=err_std_ws, marker='o', markerfacecolor='w', ls="--", lw=1.5,
                color=(255/255.0, 142/255.0, 47/255.0), elinewidth=1.0, capsize=3, capthick=1, label="SF, type II")
    ax.set_xlabel("$t$ (s)", fontsize=14)
    ax.set_ylabel("Prediction error (m)", fontsize=14)

    ax.legend(bbox_to_anchor=(0.6, 0.6, 0.5, 0.5), loc=3, fontsize=12,
              ncol=1, mode="expand", borderaxespad=0., frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(12)

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


def predict_social_force(sf_params, xr, ur, xh_init, x_goal, x_obs):
    # extract the parameters
    a = sf_params["a"]
    b = sf_params["b"]
    R = sf_params["R"]
    dv_shift = sf_params["dv_shift"]

    k = sf_params["k"]
    vd = sf_params["vd"]

    a_obs = sf_params["a_obs"]
    b_obs = sf_params["b_obs"]

    lam = 0.2

    dt = 0.05
    dt_data = 0.5
    tfac = int(dt_data / dt)
    T = (len(xr)-1) * tfac + 1

    # make an interpolation function
    t = np.arange(0, len(xr)) * dt_data
    f_xr = interpolate.interp1d(t, xr, axis=0)
    f_ur = interpolate.interp1d(t, ur, axis=0)

    # simulate with social force model
    xh = [xh_init]
    xh_prev = xh_init

    for t in range(T):
        # compute the social force to goal
        x_goal_rel = x_goal - xh_prev[0:2]
        v_goal = vd * x_goal_rel / np.linalg.norm(x_goal_rel)

        f_goal = k * (v_goal - xh_prev[2:4])

        # social force from obstacle
        x_obs_rel = xh_prev[0:2] - x_obs
        d_obs = np.linalg.norm(x_obs_rel)

        f_obs = a_obs * np.exp(-d_obs / b_obs) * (x_obs_rel / d_obs)

        cos_phi = np.dot(xh_prev[2:4], -f_obs) / np.linalg.norm(xh_prev[2:4]) / np.linalg.norm(f_obs)
        ss = lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)
        f_obs *= ss

        # social force from robot
        xr_prev = f_xr(t * dt)
        ur_prev = f_ur(t * dt)

        th = xr_prev[2]
        vr = (ur_prev[0] + dv_shift) * np.array([np.cos(th), np.sin(th)])
        v_rel = vr - xh_prev[2:4]
        x_rel = xh_prev[0:2] - xr_prev[0:2]

        if np.dot(v_rel, x_rel) <= 0:
            f_robot = np.zeros((2, ))
        else:
            d_proj = np.dot(v_rel, x_rel) * v_rel / np.linalg.norm(v_rel)**2 - x_rel
            if np.linalg.norm(d_proj) < R:
                f_robot = a * np.exp(-np.linalg.norm(d_proj) / b) * (-d_proj / np.linalg.norm(d_proj))
                cos_phi = np.dot(xh_prev[2:4], -f_robot) / np.linalg.norm(xh_prev[2:4]) / np.linalg.norm(f_robot)
                ss = lam + (1.0 - lam) * 0.5 * (1.0 + cos_phi)
                f_robot *= ss
            else:
                f_robot = np.zeros((2, ))

        # predict next step
        f_net = f_goal + f_obs + f_robot

        vh_new = xh_prev[2:4] + f_net * dt
        xh_new = xh_prev[0:2] + 0.5 * (vh_new + xh_prev[2:4]) * dt

        xh_prev = np.hstack((xh_new, vh_new))
        xh.append(xh_prev)

    return np.asarray(xh)[::tfac, :]


def predict_and_save_all_social_force(load_path, save_path, user_list, cond, ntrials, sf_params):
    t_gap = 2
    Tplan = 6

    for usr in user_list:
        print "processing user ", usr, " ..."
        path_full = load_path + "/user" + str(usr) + "/processed/" + cond

        # loading data
        init_data = np.loadtxt(path_full + "/init.txt", delimiter=",")
        goal_data = np.loadtxt(path_full + "/goal.txt", delimiter=",")
        obs_data = np.loadtxt(path_full + "/obs.txt", delimiter=',')

        for trial in range(ntrials[usr]):
            x_goal = goal_data[trial]

            file_name = path_full + "/block" + str(trial) + ".txt"
            traj = np.loadtxt(file_name, delimiter=",")

            xh = traj[:, 0:4]
            xr = traj[:, 6:9]
            ur = traj[:, 9:11]

            for t in range(0, len(xr)-Tplan, t_gap):
                xh_pred = predict_social_force(sf_params, xr[t:], ur[t:], xh[t], x_goal, obs_data)

                # plot and save the trajectory
                # fig, axes = plt.subplots()
                # plot_prediction(xh[t:], xr[t:], xh_pred, x_goal, obs_data, axes)

                # plt.show()
                file_name = save_path + "/prediction_sf/user" + str(usr) + "/" + cond + \
                            "/demo" + str(trial) + "_t" + str(t) + ".txt"
                np.savetxt(file_name, xh_pred, delimiter=',')


if __name__ == "__main__":
    load_and_plot("/home/yuhang/Documents/irl_data/winter18", 0, "hp", 0)

    # gen_init_conditions("/home/yuhang/Documents/hri_log/test_data/test_config4",
    #                     (-1, 1), (-2, 2), (0.2, 0.2), (0.5, 3.0, 0.4))

    # compute_prediction_err("/home/yuhang/Documents/irl_data/winter18",
    #                        "/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                        [0, 1, 2, 3], "hp",
    #                        [62, 27, 36, 62],
    #                        traj2d_dist_naive,
    #                        suff="_sf")
    #
    # compute_prediction_err("/home/yuhang/Documents/irl_data/winter18",
    #                        "/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                        [0, 1, 2, 3], "rp",
    #                        [62, 27, 61, 62],
    #                        traj2d_dist_naive,
    #                        suff="_sf")

    # cross_validation("/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                  "/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                  [0, 1, 2, 3], "hp",
    #                  # [0, 0, 0, 0])
    #                  [40, 17, 20, 40])

    # perform social force prediction
    # sf_params_hp = {"a": 12.0, "b": 0.608, "R": 0.178, "dv_shift": -0.108,
    #                 "k": 8.59, "vd": 1.0, "a_obs": 7., "b_obs": 2.0}
    # sf_params_rp = {"a": 12.0, "b": 0.608, "R": 0.361, "dv_shift": 0.177,
    #                 "k": 8.59, "vd": 1.0, "a_obs": 7., "b_obs": 2.0}
    #
    # predict_and_save_all_social_force("/home/yuhang/Documents/irl_data/winter18",
    #                                   "/home/yuhang/Documents/irl_data/winter18",
    #                                   [0, 1, 2, 3], "hp",
    #                                   [62, 27, 36, 62], sf_params_hp)
    #
    # predict_and_save_all_social_force("/home/yuhang/Documents/irl_data/winter18",
    #                                   "/home/yuhang/Documents/irl_data/winter18",
    #                                   [0, 1, 2, 3], "rp",
    #                                   [62, 27, 61, 62], sf_params_rp)

    # cross_validation_all_cond("/home/yuhang/Documents/irl_data/winter18/cross_validation",
    #                           [0, 1, 2, 3], [40, 17, 20, 40])
