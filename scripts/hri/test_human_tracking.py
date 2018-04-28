#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Bool
from people_msgs.msg import PositionMeasurementArray
from people_msgs.msg import People

import tty, termios, sys

from experiment_manager import getchar

# save the trajectories of all detected vehicle
trajs_people = {}
trajs_people_leg = {}

# controlling to start/stop saving
flag_start_saving = False

# how long to save
t_saving_th = 8.0


# callback functions
def people_callback(people_msg):
    if not flag_start_saving:
        return

    for person in people_msg.people:
        pos = [person.position.x, person.position.y, person.position.z]
        if person.name in trajs_people:
            trajs_people[person.name].append(pos)
        else:
            trajs_people[person.name] = [pos]


def position_arr_callback(pos_arr_msg):
    if not flag_start_saving:
        return

    for person in pos_arr_msg.people:
        pos = [person.pos.x, person.pos.y, person.pos.z]
        if person.name in trajs_people_leg:
            trajs_people_leg[person.name].append(pos)
        else:
            trajs_people_leg[person.name] = [pos]


def start_callback(msg):
    global flag_start_saving
    flag_start_saving = msg.data


if __name__ == "__main__":
    rospy.init_node("human_aware_navigation_goal_publisher")

    # subscribers
    people_sub = rospy.Subscriber("/people", People, people_callback)
    pos_arr_sub = rospy.Subscriber("/people_tracker_measurements", PositionMeasurementArray, position_arr_callback)
    start_sub = rospy.Subscriber("/test_human_tracking_start", Bool, start_callback)

    rate = rospy.Rate(50)

    while not rospy.is_shutdown():
        rate.sleep()

        while not flag_start_saving and not rospy.is_shutdown():
            rate.sleep()

        t_start = rospy.get_time()

        trajs_people.clear()
        trajs_people_leg.clear()

        print "start to save tracking!"

        while rospy.get_time() - t_start < t_saving_th and not rospy.is_shutdown():
            rate.sleep()

        # stop saving first
        flag_start_saving = False

        # plot the trajectories
        fig, axis = plt.subplots()
        for traj in trajs_people.itervalues():
            traj_plot = np.asarray(traj)
            if traj_plot.ndim == 2:
                axis.plot(traj_plot[:, 0], traj_plot[:, 1], '-o')
            else:
                axis.plot(traj_plot[0], traj_plot[1], '-o')
        axis.set_title("from people velocity tracker")

        fig, axis = plt.subplots()
        for traj in trajs_people_leg.itervalues():
            traj_plot = np.asarray(traj)
            if traj_plot.ndim == 2:
                axis.plot(traj_plot[:, 0], traj_plot[:, 1], '-o')
            else:
                axis.plot(traj_plot[0], traj_plot[1], '-o')
        axis.set_title("from leg detector")

        plt.show()
