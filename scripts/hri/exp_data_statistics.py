#!/usr/bin/env python

import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt

from plotting_utils import add_arrow

import sys
sys.path.insert(0, '../IRL')
from data_loader import wrap2pi


# far and near trials
far_trials = [6, 9, 17, 16]
near_trials = [2, 3, 11, 18]


def plot_metric_stat_box(ax, data_hp, data_rp, cond_list, metric_name):
    l_hp_mean = np.mean(data_hp, axis=0)
    l_hp_std = np.std(data_hp, axis=0)
    l_rp_mean = np.mean(data_rp, axis=0)
    l_rp_std = np.std(data_rp, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    index = np.arange(len(cond_list))
    bar_width = 0.35

    bplot1 = ax.boxplot(data_hp, positions=index, notch=False, vert=True,
                        patch_artist=True, labels=cond_list)

    bplot2 = ax.boxplot(data_rp, positions=index+bar_width, notch=False, vert=True,
                        patch_artist=True, labels=cond_list)


def plot_metric_stat_bar(ax, data_hp, data_rp, cond_list, metric_name):
    l_hp_mean = np.mean(data_hp, axis=0)
    l_hp_std = np.std(data_hp, axis=0)
    l_rp_mean = np.mean(data_rp, axis=0)
    l_rp_std = np.std(data_rp, axis=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    index = np.arange(len(cond_list))
    bar_width = 0.35

    opacity = 1.0
    error_config = {'ecolor': '0.3',
                    'capsize': 2.5,
                    'capthick': 1}

    rects1 = ax.bar(index, l_hp_mean, bar_width,
                    alpha=opacity, color=(34/255.0, 144/255.0, 196/255.0),
                    yerr=l_hp_std, error_kw=error_config,
                    label='Human Priority')

    rects2 = ax.bar(index + bar_width, l_rp_mean, bar_width,
                    alpha=opacity, color=(255/255.0, 142/255.0, 47/255.0),
                    yerr=l_rp_std, error_kw=error_config,
                    label='Robot Priority')

    ax.set_ylabel(metric_name)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(["Imp+Exp", "Imp", "Base"])

    # set y axis limit
    ax.set_ylim(0, 1.4 * np.max(np.vstack((l_rp_mean, l_hp_mean))))

    # ax.legend()


def compute_anova(data_hp, data_rp, cond_list, metric_name):
    # ANOVA for human priority trials
    st, pval = stats.f_oneway(data_hp[:, 0], data_hp[:, 1], data_hp[:, 2])
    print "Human priority trials statistics for ", metric_name, ": (F=", st, ", p=", pval, ")"

    # post-hoc test to find pairwise differences
    groups = np.tile(np.array(cond_list), (len(data_hp), 1))
    tukey = pairwise_tukeyhsd(endog=data_hp.flatten(), groups=groups.flatten(), alpha=0.05)
    # tukey.plot_simultaneous()
    print tukey.summary()

    # ANOVA for robot priority trials
    st, pval = stats.f_oneway(data_rp[:, 0], data_rp[:, 1], data_rp[:, 2])
    print "Human priority trials statistics for ", metric_name, ": (F=", st, ", p=", pval, ")"

    # post-hoc test to find pairwise differences
    tukey = pairwise_tukeyhsd(endog=data_rp.flatten(), groups=groups.flatten(), alpha=0.05)
    # tukey.plot_simultaneous()
    print tukey.summary()


def check_passing_in_front(xr, xh):
    om = 0.0

    for t in range(1, len(xr)):
        v = xh[t, 0:2] - xr[t, 0:2]
        v_last = xh[t-1, 0:2] - xr[t-1, 0:2]

        om += wrap2pi(np.arctan2(v[1], v[0]) - np.arctan2(v_last[1], v_last[0]))

    return om > 0


def passing_side_statistics_user(path, usr, cond_list, nstart=10, ntrials=20):
    # load the protocol file for each condition
    n_passing_front = []

    path += "/user" + str(usr)
    for cond in cond_list:
        n_passing_front_cond = np.zeros((2, ))
        proto = np.loadtxt(path + "/" + cond + "/" + "protocol.txt", delimiter=',')
        pp = proto[1:, 8]

        for trial in range(nstart, ntrials):
            if trial in far_trials or trial in near_trials:
                continue

            traj_data = np.loadtxt(path + "/trajectories/" + cond +
                                   "/block" + str(trial) + ".txt", delimiter=',')

            xh = traj_data[:, 0:4]
            xr = traj_data[:, 6:9]

            is_passing_in_front = check_passing_in_front(xr, xh)
            n_passing_front_cond[int(pp[trial])] += is_passing_in_front

        n_passing_front.append(n_passing_front_cond)

    return n_passing_front


def passing_side_statistics(path, usr_list, cond_list, ax=None):
    n_passing_front = []

    for usr in usr_list:
        n_passing_front.append(passing_side_statistics_user(path, usr, cond_list))

    # generate plots
    n_usr = len(usr_list)
    n_cond = len(cond_list)
    n_hp = np.zeros((n_usr, n_cond))
    n_rp = np.zeros((n_usr, n_cond))

    for i in range(n_usr):
        for k in range(n_cond):
            n_hp[i, k] = n_passing_front[i][k][0] / 3.0
            n_rp[i, k] = n_passing_front[i][k][1] / 3.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    plot_metric_stat_bar(ax, n_hp, n_rp, cond_list, "% passing in front")

    # compute statistics
    compute_anova(n_hp, n_rp, cond_list, "passing side")


def compute_safety_buffer(xr, xh):
    # relative positions
    x_rel = xh[:, 0:2] - xr[:, 0:2]
    th_rel = np.arctan2(x_rel[:, 1], x_rel[:, 0])

    for i in range(len(th_rel)):
        th_rel[i] = wrap2pi(th_rel[i] - xr[i, 2])

    d_rel = np.linalg.norm(x_rel, axis=1) / np.cos(th_rel / 2.0)
    return 1.0 / np.min(d_rel)


def compute_risk_metric(xr, xh, ur):
    risk = 0.0

    for t in range(len(xr)):
        x_rel = xh[t, 0:2] - xr[t, 0:2]
        vr = ur[t, 0] * np.array([np.cos(xr[t, 2]), np.sin(xr[t, 2])])
        v_rel = vr # - xh[t, 2:4]

        d_rel = np.linalg.norm(x_rel)
        v_proj = np.dot(v_rel, x_rel / d_rel)

        risk += max(v_proj, 0.0) / d_rel

    return risk


def compute_path_length(x, l_default=0.5, sm=0):
    path_length = 0.0

    # find direction
    ydir = x[-1, 1] - x[0, 1]

    if ydir > 0:
        em = np.argmax(x[:, 1])
    else:
        em = np.argmin(x[:, 1])

    for i in range(sm+1, em+1):
        path_length += np.linalg.norm(x[i, 0:2] - x[i-1, 0:2])

    l_straight = np.linalg.norm(x[em, 0:2] - x[sm, 0:2])
    # return path_length - l_straight + l_default
    return path_length - 6.0


def compute_effort(uh):
    return np.sum(np.linalg.norm(uh, axis=1))


def safety_efficiency_user(path, usr, cond_list, nstart=10, ntrials=20):
    safety_metric = []
    path_length = []
    effort = []
    risk = []

    path += "/user" + str(usr)
    for cond in cond_list:
        safety_cond = [[], []]
        path_length_cond = [[], []]
        effort_cond = [[], []]
        risk_cond = [[], []]

        proto = np.loadtxt(path + "/" + cond + "/" + "protocol.txt", delimiter=',')
        pp = proto[1:, 8]

        for trial in range(nstart, ntrials):
            if trial in far_trials or trial in near_trials:
                continue

            traj_data = np.loadtxt(path + "/trajectories/" + cond +
                                   "/block" + str(trial) + ".txt", delimiter=',')

            xh = traj_data[:, 0:4]
            uh = traj_data[:, 4:6]
            xr = traj_data[:, 6:9]
            ur = traj_data[:, 9:11]

            safety_cond[int(pp[trial])].append(compute_safety_buffer(xr, xh))
            path_length_cond[int(pp[trial])].append(compute_path_length(xh))
            effort_cond[int(pp[trial])].append(compute_effort(uh))
            risk_cond[int(pp[trial])].append(compute_risk_metric(xr, xh, ur))

        safety_metric.append(safety_cond)
        path_length.append(path_length_cond)
        effort.append(effort_cond)
        risk.append(risk_cond)

    return safety_metric, path_length, effort, risk


def compute_efficiency(xr, xh, ur, th=2.0):
    efficiency = 0.0

    count = 0.0
    wh = 1.0
    wr = 0.0
    for t in range(len(xr)):
        x_rel = xh[t, 0:2] - xr[t, 0:2]

        if np.linalg.norm(x_rel) < th:
            efficiency += wr * ur[t, 0] + wh * np.linalg.norm(xh[t, 2:4])
            # efficiency += np.linalg.norm(xh[t, 2:4]) / np.linalg.norm(x_rel)
            count += 1.0
            # count += np.linalg.norm(xh[t, 0:2] - xh[t-1, 0:2])

    if count == 0:
        return compute_efficiency(xr, xh, ur, th+0.5)
    else:
        return efficiency / count


def compute_efficiency_user(path, usr, cond_list, nstart=10, ntrials=20):
    efficiency = []

    path += "/user" + str(usr)
    for cond in cond_list:
        efficiency_cond = [[], []]

        proto = np.loadtxt(path + "/" + cond + "/" + "protocol.txt", delimiter=',')
        pp = proto[1:, 8]

        for trial in range(nstart, ntrials):
            if trial in far_trials or trial in near_trials:
                continue

            traj_data = np.loadtxt(path + "/trajectories/" + cond +
                                   "/block" + str(trial) + ".txt", delimiter=',')

            xh = traj_data[:, 0:4]
            uh = traj_data[:, 4:6]
            xr = traj_data[:, 6:9]
            ur = traj_data[:, 9:11]

            efficiency_cond[int(pp[trial])].append(compute_efficiency(xr, xh, ur))

        efficiency.append(efficiency_cond)

    return efficiency


def compute_and_plot_stats(path, usr_list, cond_list, ax=None):
    safety_metric = [[], []]
    path_length = [[], []]
    human_effort = [[], []]
    risk = [[], []]
    efficiency = [[], []]

    for usr in usr_list:
        safety_user, path_length_user, effort_user, risk_user = safety_efficiency_user(path, usr, cond_list)
        efficiency_user = compute_efficiency_user(path, usr, cond_list)

        if not safety_metric[0]:
            for i in range(len(cond_list)):
                safety_metric[0].append(safety_user[i][0])
                safety_metric[1].append(safety_user[i][1])
                path_length[0].append(path_length_user[i][0])
                path_length[1].append(path_length_user[i][1])
                human_effort[0].append(effort_user[i][0])
                human_effort[1].append(effort_user[i][1])
                risk[0].append(risk_user[i][0])
                risk[1].append(risk_user[i][1])
                efficiency[0].append(efficiency_user[i][0])
                efficiency[1].append(efficiency_user[i][1])
        else:
            for i in range(len(cond_list)):
                safety_metric[0][i] += safety_user[i][0]
                path_length[0][i] += path_length_user[i][0]
                human_effort[0][i] += effort_user[i][0]
                risk[0][i] += risk_user[i][0]
                safety_metric[1][i] += safety_user[i][1]
                path_length[1][i] += path_length_user[i][1]
                human_effort[1][i] += effort_user[i][1]
                risk[1][i] += risk_user[i][1]

                efficiency[0][i] += efficiency_user[i][0]
                efficiency[1][i] += efficiency_user[i][1]

    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot(path_length[0][0], 'x')
    # axes[1].plot(path_length[0][1], 'x')
    # plt.show()

    # for priority in range(2):
    #     fig, axes = plt.subplots(1, 2)
    #
    #     safety_plot = np.array(safety_metric[priority])
    #     path_length_plot = np.array(path_length[priority])
    #     human_effort_plot = np.array(human_effort[priority])
    #
    #     colors = ['r', 'b', 'k']
    #     for i in range(len(cond_list)):
    #         axes[0].plot(safety_plot[i], path_length_plot[i], 'x', color=colors[i], label=cond_list[i])
    #         axes[1].plot(safety_plot[i], human_effort_plot[i], 'x', color=colors[i], label=cond_list[i])
    #
    #     axes[0].legend()
    #     axes[1].legend()

    n_cond = len(cond_list)

    l_hp_trial = np.asarray(path_length[0]).transpose()
    l_rp_trial = np.asarray(path_length[1]).transpose()

    l_hp = np.zeros((len(usr_list), n_cond))
    l_rp = np.zeros_like(l_hp)

    for i in range(len(usr_list)):
        for j in range(n_cond):
            l_hp[i] += l_hp_trial[i*n_cond+j] / float(n_cond)
            l_rp[i] += l_rp_trial[i*n_cond+j] / float(n_cond)

    # l_hp_mean = np.mean(path_length[0], axis=1)
    # l_hp_std = np.std(path_length[0], axis=1)
    # l_rp_mean = np.mean(path_length[1], axis=1)
    # l_rp_std = np.std(path_length[1], axis=1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    plot_metric_stat_bar(ax, l_hp, l_rp, cond_list, "path length")

    compute_anova(l_hp, l_rp, cond_list, "path length")


def plot_all_stats(path, usr_list, cond_list):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    passing_side_statistics(path, usr_list, cond_list, ax=axes[0])
    compute_and_plot_stats(path, usr_list, cond_list, ax=axes[1])

    # compute and plot trust
    trust_data = np.array([[6, 4, 3],
                           [7, 4, 3],
                           [7, 5, 3],
                           [7, 6, 5],
                           [6, 4, 4],
                           [7, 5, 3],
                           [6, 4, 2],
                           [7, 5, 4],
                           [7, 5, 4],
                           [6, 4, 4],
                           [6, 4, 2],
                           [7, 5, 3]], dtype=float)

    trust_mean = np.mean(trust_data, axis=0)
    trust_std = np.std(trust_data, axis=0)

    index = np.arange(len(cond_list))
    bar_width = 0.6

    opacity = 1.0
    error_config = {'ecolor': '0.3',
                    'capsize': 2.5,
                    'capthick': 1}

    rects = axes[2].bar(index, trust_mean, bar_width,
                        alpha=opacity, color=(3/255.0, 125/255.0, 189/255.0),
                        yerr=trust_std, error_kw=error_config) #, hatch="//", edgecolor=(255/255.0, 142/255.0, 47/255.0))
    rects[1].set_color((71/255.0, 164/255.0, 212/255.0))
    # rects[1].set_hatch("//")
    # rects[1].set_edgecolor((255/255.0, 142/255.0, 47/255.0))
    rects[2].set_color((157/255.0, 205/255.0, 232/255.0))
    # rects[2].set_hatch("//")
    # rects[2].set_edgecolor((255/255.0, 142/255.0, 47/255.0))

    axes[2].set_ylabel('trust')
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(["Imp+Exp", "Imp", "Base"])

    # ANOVA for trust
    st, pval = stats.f_oneway(trust_data[:, 0], trust_data[:, 1], trust_data[:, 2])
    print "Statistics for trust: (F=", st, ", p=", pval, ")"

    # post-hoc test to find pairwise differences
    groups = np.tile(np.array(cond_list), (len(trust_data), 1))
    tukey = pairwise_tukeyhsd(endog=trust_data.flatten(), groups=groups.flatten(), alpha=0.05)
    # tukey.plot_simultaneous()
    print tukey.summary()

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    cond_list = ["haptics", "no_haptics", "baseline"]  # , "baseline"]

    # passing_side_statistics("/home/yuhang/Documents/hri_log/exp_data",
    #                         [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], cond_list)
    # compute_and_plot_stats("/home/yuhang/Documents/hri_log/exp_data",
    #                        [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], cond_list)
    # plt.show()

    plot_all_stats("/home/yuhang/Documents/hri_log/exp_data",
                   [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12], cond_list)
