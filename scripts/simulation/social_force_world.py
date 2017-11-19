#!/usr/bin/env python

from os import path
import sys
sys.path.append(path.abspath("../../external/planning_algorithms"))

import rospy
import numpy as np
import matplotlib.pyplot as plt
import json

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D

from environment.worlds.occupancy_grid import OccupancyGrid
from environment.robots.turtlebot import Turtlebot
from environment.robots.humans import HumanSimple


class SFWorld:
    def __init__(self):
        # load a map
        map_path = rospy.get_param("~map_path",
                                   "../../external/planning_algorithms/resources/occ_maps/test_map.png")
        map_res = rospy.get_param("~map_resolution", 0.01)

        self.grid_map = OccupancyGrid(map_path, map_res)
        map_width = self.grid_map.width * map_res
        map_height = self.grid_map.height * map_res

        # load agent settings
        agent_setting_file = rospy.get_param("~agent_setting_file",
                                             "../../resources/sim_setting/default.json")
        settings = json.load(open(agent_setting_file))

        self.nhuman = settings["num_humans"]

        self.humans = []
        self.human_param = []
        for k in range(self.nhuman):
            human_id = "human" + str(k)
            goal = np.array([settings[human_id]["goal_x"], settings[human_id]["goal_y"]])
            k = settings[human_id]["k"]
            vd = settings[human_id]["vd"]

            human = HumanSimple()
            human.set_goal(goal, k, vd)
            self.humans.append(human)

            # load the social force parameters
            ah = settings[human_id]["hh_param"][0]
            bh = settings[human_id]["hh_param"][1]
            Rh = settings[human_id]["hh_param"][2]

            ar = settings[human_id]["hr_param"][0]
            br = settings[human_id]["hr_param"][1]
            Rr = settings[human_id]["hr_param"][2]

            self.human_param.append((ah, bh, Rh, ar, br, Rr))

            # start poses
            pose_start = np.array(settings[human_id]["pose_start"])
            human.set_pose(pose_start)

        # create a robot
        self.robot = Turtlebot()

        # set robot initial position
        pose_start = np.array(settings["robot"]["pose_start"])
        self.robot.set_pose(pose_start)

        # robot command input
        self.cmd_vel = np.array([0.0, 0.0])

        # publishers and subscribers
        self.cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        self.human_pubs = []
        for k in range(self.nhuman):
            topic_name = "/human_poses/human" + str(k)
            self.human_pubs.append(rospy.Publisher(topic_name, Pose2D, queue_size=1))

        # create a plot axis
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, map_width)
        self.ax.set_ylim(0, map_height)
        plt.ion()

        # self.grid_map.visualize(self.ax)

        # visualize human and robot
        for human in self.humans:
            human.visualize(self.ax)

        self.robot.visualize(self.ax)

        # initialize a counter
        self.counter = 0

    def cmd_vel_callback(self, cmd_vel_msg):
        self.cmd_vel[0] = cmd_vel_msg.linear.x
        self.cmd_vel[1] = cmd_vel_msg.angular.z

    def update(self, dt):
        self.counter += 1

        for human in self.humans:
            human.update(np.zeros((2,)), dt)
            if self.counter % 5 == 1:
                human.visualize(self.ax)

        if self.counter % 5 == 1:
            plt.pause(0.001)


if __name__ == "__main__":
    # initialize node
    rospy.init_node("simulation_world")

    # create the world object
    world = SFWorld()

    # loop
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        print "hhh"
        world.update(0.01)
        rate.sleep()
