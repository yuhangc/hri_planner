#!/usr/bin/env python

import numpy as np

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from hri_planner.msg import PlannedTrajectories


class PlannerTester(object):
    def __init__(self):
        # dimensions
        self.T_ = rospy.get_param("~dimension/T", 10)
        self.nXh_ = rospy.get_param("~dimension/nXh", 4)
        self.nUh_ = rospy.get_param("~dimension/nUh", 2)
        self.nXr_ = rospy.get_param("~dimension/nXr", 3)
        self.nUr_ = rospy.get_param("~dimension/nUr", 2)
        self.dt_ = rospy.get_param("~dimension/dt", 0.5)

        # things to publish
        self.robot_state_ = np.zeros((self.nXr_, ))
        self.robot_vel_ = np.zeros((self.nUr_, ))
        self.human_state_ = np.zeros((self.nXh_, ))
        self.human_acc_ = np.zeros((self.nUh_, ))

        # things to subscribe to
        self.robot_traj_opt_ = None
        self.human_traj_hp_opt_ = None
        self.human_traj_rp_opt_ = None

        self.acomm_ = -1

        self.robot_ctrl_ = np.zeros((self.nUr_, ))

        # "actual" trajectories and initial states
        self.xr0 = np.zeros((self.nXr_, ))
        self.xh0 = np.zeros((self.nXh_, ))
        self.robot_traj = []
        self.human_traj = []

        # goals
        self.xr_goal = np.zeros((2, ))
        self.xh_goal = np.zeros((2, ))

    def load_data(self, path):
        # load initial states, human trajectory, goal states
        pass

    # callbacks
    def comm_callback(self, comm_msg):
        self.acomm_ = comm_msg.data

    def robot_ctrl_callback(self, ctrl_msg):
        self.robot_ctrl_[0] = ctrl_msg.data[0]
        self.robot_ctrl_[1] = ctrl_msg.data[1]

    def plan_callback(self, plan_msg):
        msg = PlannedTrajectories

        self.robot_traj_opt_ = np.asarray(plan_msg.robot_traj_opt)
        self.human_traj_hp_opt_ = np.asarray(plan_msg.human_traj_hp_opt_)
        self.human_traj_rp_opt_ = np.asarray(plan_msg.human_traj_rp_opt_)

        pass
