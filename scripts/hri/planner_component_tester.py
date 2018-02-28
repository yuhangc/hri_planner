#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import rospy

from hri_planner.srv import *


def belief_update_client(xr, ur, xh0, acomm, tcomm, t_total):
    rospy.wait_for_service("update_belief")
    try:
        belief_update = rospy.ServiceProxy("update_belief", BeliefUpdate)
        beliefs = belief_update(xr, ur, xh0, acomm, tcomm, t_total)

        return beliefs
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e


def test_belief_update(user_id, cond, trial, acomm, tcomm):
    # load data
    path = rospy.get_param("~data_path", "/home/yuhang/Documents/irl_data/winter18")
    path += "/user" + str(user_id) + "/processed/" + cond + "/block" + str(trial) + ".txt"

    traj_data = np.loadtxt(path, delimiter=',')
    xh = traj_data[:, 0:4]
    uh = traj_data[:, 4:6]
    xr = traj_data[:, 6:9]
    ur = traj_data[:, 9:11]

    # calculate belief
    beliefs = belief_update_client(xr.flatten(), ur.flatten(),
                                   xh.flatten(), acomm, tcomm, len(xr))
    beliefs = np.asarray(beliefs.belief)

    # plotting
    fig, axes = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(xh[:, 0], xh[:, 1], '--ok', lw=1, fillstyle="none", label="human")
    axes[0].plot(xr[:, 0], xr[:, 1], '-or', lw=1, fillstyle="none", label="robot")
    axes[0].plot(xr[-1, 0], xr[-1, 1], 'or', markersize=10)
    axes[0].plot(xh[-1, 0], xh[-1, 1], 'ok', markersize=10)
    axes[0].axis("equal")
    axes[0].legend()

    axes[1].plot(beliefs, '-ob', lw=1, fillstyle="none")

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

if __name__ == "__main__":
    # no communication
    test_belief_update(0, "hp", 0, 0, -100)
    # communication matches action
    test_belief_update(0, "hp", 0, 0, 0)
    # communication doesn't match action
    test_belief_update(0, "hp", 0, 1, 0)

    # robot priority scenarios
    test_belief_update(0, "rp", 0, 0, -100)
    test_belief_update(0, "rp", 0, 0, 0)
    test_belief_update(0, "rp", 0, 1, 0)
