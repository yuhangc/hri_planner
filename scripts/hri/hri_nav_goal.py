#!/usr/bin/env python

import numpy as np

import rospy
import actionlib
import move_base_msgs.msg
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

import tty, termios, sys

flag_manual_goal = False


def rviz_goal_callback(goal_msg):
    # if not manual mode do nothing
    if not flag_manual_goal:
        return

    # create a navigation goal
    goal = move_base_msgs.msg.MoveBaseGoal()
    goal.target_pose = goal_msg

    # send through actionlib client
    hri_nav_client.send_goal(goal)


def load_navigation_goals(n_goals):
    nav_goals = []
    for i in range(n_goals):
        param_name = "~goal" + str(i)

        # create a navigation goal
        goal = PoseStamped()
        goal.pose.position.x = rospy.get_param(param_name + "/x")
        goal.pose.position.y = rospy.get_param(param_name + "/y")
        goal.pose.position.z = 0.0

        th = rospy.get_param(param_name + "/theta", 0.0)
        goal.pose.orientation.z = np.sin(th / 2.0)
        goal.pose.orientation.w = np.cos(th / 2.0)

        nav_goals.append(goal)

        rospy.loginfo("loaded goal %d, (%f, %f, %f)", i, goal.pose.position.x,
                      goal.pose.position.y, th)

    return nav_goals


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


if __name__ == "__main__":
    rospy.init_node("human_aware_navigation_goal_publisher")

    # get ros parameters about patrolling goals and mode
    n_goals = rospy.get_param("~num_goals", 0)

    rospy.loginfo("Loading %d navigation goals", n_goals)
    nav_goals = load_navigation_goals(n_goals)

    flag_manual_goal = rospy.get_param("~manual_mode", False)

    # create a subscriber to goal published by rviz
    rospy.Subscriber(
        "/rviz_navigation_goal",
        PoseStamped,
        rviz_goal_callback
    )

    # create an actionlib client
    rospy.loginfo("Creating human aware navigation client.")
    hri_nav_client = actionlib.SimpleActionClient(
        'human_aware_navigation',
        move_base_msgs.msg.MoveBaseAction
    )
    hri_nav_client.wait_for_server()
    rospy.loginfo("...done")

    # two modes
    if flag_manual_goal:
        # start
        rospy.spin()
    else:
        rate = rospy.Rate(20)
        # first wait for a keyboard input
        print "Please press 's' to start:"
        while getchar() != 's':
            rate.sleep()
        rospy.loginfo("started!")

        # start looping through all goals
        id = 0
        flag_goal_reached = True

        while not rospy.is_shutdown():
            # send a navigation goal
            goal = move_base_msgs.msg.MoveBaseGoal()
            goal.target_pose = nav_goals[id]
            hri_nav_client.send_goal(goal)

            # wait for result
            hri_nav_client.wait_for_result()

            # get result
            res = hri_nav_client.get_result()

            if res == GoalStatus.SUCCEEDED:
                # increase id and reset flag
                id += 1
                if id == n_goals:
                    id = 0

                flag_goal_reached = False
            else:
                # cancel and resend goal
                hri_nav_client.cancel_all_goals()

            rate.sleep()
