#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import rospy
from hri_planner.srv import *

from plotting_utils import add_arrow


def belief_update_client(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path):
    rospy.wait_for_service("update_belief")
    try:
        belief_update = rospy.ServiceProxy("update_belief", TestComponent)
        succeeded = belief_update(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path)

        return succeeded
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def cost_feature_test_client(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path):
    rospy.wait_for_service("test_cost_features")
    try:
        test_features = rospy.ServiceProxy("test_cost_features", TestComponent)
        succeeded = test_features(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path)

        return succeeded
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def simple_optimizer_client(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path):
    rospy.wait_for_service("test_simple_optimizer")
    try:
        test_optimizer = rospy.ServiceProxy("test_simple_optimizer", TestComponent)
        succeeded = test_optimizer(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path)

        return succeeded
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def prob_cost_client(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path):
    rospy.wait_for_service("test_prob_cost")
    try:
        test_prob_cost = rospy.ServiceProxy("test_prob_cost", TestComponent)
        succeeded = test_prob_cost(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path)

        return succeeded
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def nested_optimizer_client(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path):
    rospy.wait_for_service("test_nested_optimizer")
    try:
        test_optimizer = rospy.ServiceProxy("test_nested_optimizer", TestComponent)
        succeeded = test_optimizer(xh, uh, xr, ur, xh0, xr0, weights, acomm, tcomm, log_path)

        return succeeded
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def test_belief_update(user_id, cond, trial, acomm, tcomm, flag_plot=True):
    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond

    traj_data = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=',')
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")

    xh = traj_data[:, 0:4]
    uh = traj_data[:, 4:6]
    xr = traj_data[:, 6:9]
    ur = traj_data[:, 9:11]
    xh0 = init_data[trial, 0:4]
    xr0 = init_data[trial, 4:7]

    # calculate belief
    log_path = "/home/yuhang/Documents/hri_log"
    succeeded = belief_update_client(xh.flatten(), uh.flatten(),
                                     xr.flatten(), ur.flatten(), xh0, xr0, None,
                                     acomm, tcomm, log_path)

    # load the calculated belief
    belief_data = np.loadtxt(log_path + "/log_belief.txt")

    # plotting
    if flag_plot:
        fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
        axes[0].plot(xh[:, 0], xh[:, 1], '--ok', lw=1, fillstyle="none", label="human")
        axes[0].plot(xr[:, 0], xr[:, 1], '-or', lw=1, fillstyle="none", label="robot")
        axes[0].plot(xr[-1, 0], xr[-1, 1], 'or', markersize=10)
        axes[0].plot(xh[-1, 0], xh[-1, 1], 'ok', markersize=10)
        axes[0].axis("equal")
        axes[0].legend()

        # axes[1].plot(belief_data[0], '-ob', lw=1, fillstyle="none")
        # axes[1].plot(belief_data[1], '--xk', lw=1, fillstyle="none")
        axes[1].plot(belief_data, '-sb', lw=1.5, fillstyle="none")

        if cond == "hp":
            title = "human priority"
        else:
            title = "robot priority"

        if tcomm < 0:
            title += ", no explicit communication"
        else:
            if acomm == 0:
                title += ", comm. is: human priority, t = " + str(tcomm)
            else:
                title += ", comm. is: robot priority, t = " + str(tcomm)

        plt.suptitle(title, fontsize=16)
        plt.show()

    return belief_data


def plot_belief_update_examples(user_id, cond, trial, t_plot):
    # perform simulation for all different communication options
    belief_data_nc = test_belief_update(user_id, cond, trial, 0, -20, flag_plot=False)
    belief_data_hp = test_belief_update(user_id, cond, trial, 0, 1, flag_plot=False)
    belief_data_rp = test_belief_update(user_id, cond, trial, 1, 1, flag_plot=False)

    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond

    traj_data = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=',')
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")

    xh = traj_data[:, 0:4]
    uh = traj_data[:, 4:6]
    xr = traj_data[:, 6:9]
    ur = traj_data[:, 9:11]
    xh0 = init_data[trial, 0:4]
    xr0 = init_data[trial, 4:7]

    # create plots
    fig = plt.figure(figsize=(7.5, 5))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.5, 1])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[1, :])
    # ax4 = plt.subplot(gs[2, :])
    # ax5 = plt.subplot(gs[3, :])

    # ax0 = plt.subplot2grid((4, 3), (0, 0), rowspan=1)
    # ax1 = plt.subplot2grid((4, 3), (0, 1), rowspan=1)
    # ax2 = plt.subplot2grid((4, 3), (0, 2), rowspan=1)
    # ax3 = plt.subplot2grid((4, 3), (1, 0), colspan=3)
    # ax4 = plt.subplot2grid((4, 3), (2, 0), colspan=3)
    # ax5 = plt.subplot2grid((4, 3), (3, 0), colspan=3)

    # plot the trajectories
    visualize_frame(ax0, xh, xr, t_plot[0], "/home/yuhang/Documents/exp_maps/atrium0201/my_map.jpg")
    visualize_frame(ax1, xh, xr, t_plot[1], "/home/yuhang/Documents/exp_maps/atrium0201/my_map.jpg")
    visualize_frame(ax2, xh, xr, t_plot[2], "/home/yuhang/Documents/exp_maps/atrium0201/my_map.jpg")

    # plot beliefs
    ax3.plot(belief_data_nc, '-s', lw=2, color=(0.2, 0.2, 0.6), markersize=7, fillstyle="none", label="no comm.")
    ax3.plot(belief_data_hp, '-o', lw=2, color=(0.0, 0.0, 0.2), markersize=7, fillstyle="none", label="h.p.")
    ax3.plot(belief_data_rp, '-^', lw=2, color=(0.4, 0.4, 0.9), markersize=7, fillstyle="none", label="r.p.")
    ax3.axis([0, 16, -0.1, 1.5])
    ax3.set_yticks(np.arange(0, 1.1, 0.5))
    ax3.legend(ncol=3, fontsize=12)
    # ax4.axis([0, 16, -0.2, 1.2])
    # ax5.axis([0, 16, -0.2, 1.2])

    plt.show()


def visualize_frame(ax, xh, xr, t, map_path):
    img = plt.imread(map_path)
    usize = 80
    xstart = 400
    ystart = 280
    xend = xstart + 4.5 * usize
    yend = ystart + 7.5 * usize

    field_range = [-0.5, 5, -0.5, 8]

    ax.imshow(img[ystart:yend, xstart:xend], extent=field_range, cmap="gray")
    # ax.plot(xh[:t-2, 0], xh[:t-2, 1], '-', color=(0.8, 0.8, 0.8), lw=1.5)
    # ax.plot(xr[:t-2, 0], xr[:t-2, 1], '-', color=(0.8, 0.8, 0.8), lw=1.5)

    human_traj = ax.plot(xh[:t+1, 0], xh[:t+1, 1], '-k', lw=1.5, label="human")
    add_arrow(human_traj[0], position=xh[t-1, 0], size=15)
    robot_traj = ax.plot(xr[:t+1, 0], xr[:t+1, 1], '-r', lw=1.5, label="robot")
    add_arrow(robot_traj[0], position=xr[t-1, 0], size=15)

    ax.axis("equal")
    ax.axis(field_range)

    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off

    ax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off') # labels along the bottom edge are off


def test_cost_features(user_id, cond, trial):
    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond

    traj_data = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=',')
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")

    T = 10
    xh = traj_data[:T, 0:4]
    uh = traj_data[:T, 4:6]
    xr = traj_data[:T, 6:9]
    ur = traj_data[:T, 9:11]
    xh0 = init_data[trial, 0:4]
    xr0 = init_data[trial, 4:7]

    # calculate belief
    log_path = "/home/yuhang/Documents/hri_log"
    succeeded = cost_feature_test_client(xh.flatten(), uh.flatten(), xr.flatten(),
                                         ur.flatten(), xh0.flatten(), xr0.flatten(),
                                         None, None, None, log_path)

    # TODO: some visualization?
    # traj_data = np.loadtxt(log_path + "/log_traj.txt")
    # xr_pred = traj_data[:T*3].reshape(T, 3)
    #
    # fig, axes = plt.subplots()
    # axes.plot(xr_pred[:, 0], xr_pred[:, 1], '-or')
    # axes.plot(xr[:, 0], xr[:, 1], '-bo')
    # axes.axis("equal")
    # plt.show()


def test_simple_optimizer(user_id, cond, trial):
    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond

    traj_data = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=',')
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")

    T = 10
    xh = traj_data[:T, 0:4]
    uh = traj_data[:T, 4:6]
    xr = traj_data[:T, 6:9]
    ur = traj_data[:T, 9:11]
    xh0 = init_data[trial, 0:4]
    xr0 = init_data[trial, 4:7]

    # calculate belief
    log_path = "/home/yuhang/Documents/hri_log"
    succeeded = simple_optimizer_client(xh.flatten(), uh.flatten(), xr.flatten(),
                                        ur.flatten(), xh0.flatten(), xr0.flatten(),
                                        np.array([8.0, 20.0, 50.0, 7.0, 2.0]),
                                        None, None, log_path)

    # TODO: some visualization?
    traj_data = np.loadtxt(log_path + "/log_traj.txt")
    xh_pred = traj_data[:T*4].reshape(T, 4)

    fig, axes = plt.subplots()
    axes.plot(xh_pred[:, 0], xh_pred[:, 1], '-ok')
    axes.plot(xh[:, 0], xh[:, 1], '-bo')
    axes.axis("equal")
    plt.show()


def test_prob_cost(user_id, cond, trial):
    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond

    traj_data = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=',')
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")

    T = 10
    xh = traj_data[:T, 0:4]
    uh = traj_data[:T, 4:6]
    xr = traj_data[:T, 6:9]
    ur = traj_data[:T, 9:11]
    xh0 = init_data[trial, 0:4]
    xr0 = init_data[trial, 4:7]

    # calculate belief
    log_path = "/home/yuhang/Documents/hri_log"
    succeeded = prob_cost_client(xh.flatten(), uh.flatten(), xr.flatten(),
                                 ur.flatten(), xh0.flatten(), xr0.flatten(),
                                 np.array([1.0, 1.0, 1.0, 1.0]),
                                 None, None, log_path)


def test_nested_optimizer(user_id, cond, trial, acomm, tcomm):
    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond

    traj_data = np.loadtxt(path + "/block" + str(trial) + ".txt", delimiter=',')
    init_data = np.loadtxt(path + "/init.txt", delimiter=",")

    T = 10
    xh = traj_data[:T, 0:4]
    uh = traj_data[:T, 4:6]
    xr = traj_data[:T, 6:9]
    ur = traj_data[:T, 9:11]
    xh0 = init_data[trial, 0:4]
    xr0 = init_data[trial, 4:7]

    # calculate belief
    log_path = "/home/yuhang/Documents/hri_log"
    succeeded = nested_optimizer_client(xh.flatten(), uh.flatten(), xr.flatten(),
                                        ur.flatten(), xh0.flatten(), xr0.flatten(),
                                        np.array([8.0, 20.0, 40.0, 7.0, 1.0,
                                                  8.0, 20.0, 40.0, 7.0, 10.0,
                                                  10.0, 20.0, 5.0, 5.0]),
                                        acomm, tcomm, log_path)

    # TODO: some visualization?
    human_traj_data = np.loadtxt(log_path + "/log_human_traj.txt")
    robot_traj_data = np.loadtxt(log_path + "/log_robot_traj.txt")

    # xh_pred_hp = human_traj_data[0].reshape(T, 4)
    # xh_pred_rp = human_traj_data[1].reshape(T, 4)
    # xh_opt_hp = human_traj_data[2].reshape(T, 4)
    # xh_opt_rp = human_traj_data[3].reshape(T, 4)
    xh_opt_hp = human_traj_data[0].reshape(T, 4)
    xh_opt_rp = human_traj_data[1].reshape(T, 4)
    xr_opt = robot_traj_data.reshape(T, 3)

    fig, axes = plt.subplots()
    # axes.plot(xh_pred_hp[:, 0], xh_pred_hp[:, 1], '-ok')
    # axes.plot(xh_pred_rp[:, 0], xh_pred_rp[:, 1], '--ok')
    axes.plot(xh_opt_hp[:, 0], xh_opt_hp[:, 1], '-xr')
    axes.plot(xh_opt_rp[:, 0], xh_opt_rp[:, 1], '--xr')
    axes.plot(xh[:, 0], xh[:, 1], '-', color="grey")
    #
    axes.plot(xr[:, 0], xr[:, 1], 'b-')
    axes.plot(xr_opt[:, 0], xr_opt[:, 1], '-ob')

    axes.axis("equal")
    plt.show()


if __name__ == "__main__":
    # # no communication
    # test_belief_update(0, "hp", 0, 0, -20)
    # # communication matches action
    # test_belief_update(0, "hp", 0, 0, 0)
    # # communication doesn't match action
    # test_belief_update(0, "hp", 0, 1, 0)
    #
    # # robot priority scenarios
    # test_belief_update(0, "rp", 0, 0, -20)
    # test_belief_update(0, "rp", 0, 0, 0)
    # test_belief_update(0, "rp", 0, 1, 0)

    plot_belief_update_examples(0, "hp", 0, [4, 7, 14])

    # # test the cost features
    # test_cost_features(0, "hp", 0)

    # test the simple optimizer
    # test_simple_optimizer(0, "hp", 0)

    # test the probabilistic cost
    # test_prob_cost(0, "hp", 0)

    # test the nested optimizer
    # test_nested_optimizer(0, "hp", 0, 0, -20)
