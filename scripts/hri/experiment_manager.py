#!/usr/bin/env python

import numpy as np

import rospy
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import tty, termios, sys


def getchar():
    # Returns a single character from standard input
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


class ExperimentManager(object):
    def __init__(self):
        self.comm_sub = rospy.Subscriber("/planner/communication", String, self.comm_callback)

        # load goals
        protocol_file = rospy.get_param("~protocol_file", "../../resources/exp_protocols/protocol.txt")

        self.xr_goal = []
        self.xh_goal = []
        self.xh_init = []
        self.intent = []

        self.load_goals(protocol_file)

        self.acomm = ""
        self.flag_comm_updated = False

        self.flag_goal_reached = True

        # subscribers and publishers
        self.goal_reach_sub = rospy.Subscriber("planner/goal_reached", Bool, self.goal_reached_callback)

        self.goal_pub = rospy.Publisher("/planner/set_goal", Float64MultiArray, queue_size=1)
        self.planner_ctrl_pub = rospy.Publisher("/planner/ctrl", String, queue_size=1)

    def load_goals(self, protocol_file):
        proto_data = np.loadtxt(protocol_file, delimiter=',')
        print proto_data

        # parse the protocol file
        self.xr_goal = proto_data[:, 1:3]
        self.xh_goal = proto_data[:, 3:5]
        self.xh_init = proto_data[:, 5:7]
        self.intent = proto_data[:, 7]

    def run(self, trial_start=0):
        rate = rospy.Rate(20)

        trial = trial_start
        while not rospy.is_shutdown():
            rate.sleep()

            # publish new goal if goal reached
            if self.flag_goal_reached:
                if trial >= len(self.xr_goal):
                    print "trials ended!"
                    break

                self.flag_goal_reached = False
                self.publish_goal(trial)
                trial += 1

                print "Please press 's' to start:"
                while getchar() != 's':
                    rate.sleep()

                # tell planner to start
                ctrl_data = String()
                ctrl_data.data = "start"
                self.planner_ctrl_pub.publish(ctrl_data)

                rospy.loginfo("started!")

    def publish_goal(self, trial):
        goal_data = Float64MultiArray()
        for xr in self.xr_goal[trial]:
            goal_data.data.append(xr)
        for xh in self.xh_goal[trial]:
            goal_data.data.append(xh)
        for xh in self.xh_init[trial]:
            goal_data.data.append(xh)

        # set intent data
        goal_data.data.append(self.intent[trial])
        print goal_data.data

        self.goal_pub.publish(goal_data)
        print "goal data sent!"

    def comm_callback(self, comm_msg):
        if comm_msg.data == "Attract":
            self.acomm = "HumanPriority"
        else:
            self.acomm = "RobotPriority"

        self.flag_comm_updated = True
        print "received communication: ", self.acomm

    def goal_reached_callback(self, msg):
        self.flag_goal_reached = msg.data
        print "Goal reached: ", msg.data


if __name__ == "__main__":
    rospy.init_node("human_aware_navigation_goal_publisher")

    exp_manager = ExperimentManager()
    exp_manager.run(0)
