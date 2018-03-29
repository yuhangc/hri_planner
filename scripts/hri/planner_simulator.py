#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from hri_planner.msg import PlannedTrajectories


class PlannerSimulator(object):
    def __init__(self):
        # dimensions
        self.T_ = rospy.get_param("~dimension/T", 10)
        self.nXh_ = rospy.get_param("~dimension/nXh", 4)
        self.nUh_ = rospy.get_param("~dimension/nUh", 2)
        self.nXr_ = rospy.get_param("~dimension/nXr", 3)
        self.nUr_ = rospy.get_param("~dimension/nUr", 2)
        self.dt_ = rospy.get_param("~dimension/dt", 0.5)

        # things to publish
        self.xr_ = np.zeros((self.nXr_, ))
        self.ur_ = np.zeros((self.nUr_, ))
        self.xh_ = np.zeros((self.nXh_, ))
        self.uh_ = np.zeros((self.nUh_, ))

        # things to subscribe to
        self.robot_traj_opt_ = None
        self.human_traj_hp_opt_ = None
        self.human_traj_rp_opt_ = None

        self.acomm_ = -1
        self.robot_intent_ = -1

        self.robot_ctrl_ = np.zeros((self.nUr_, ))

        # "actual" trajectories and initial states
        self.xr0 = np.zeros((self.nXr_, ))
        self.xh0 = np.zeros((self.nXh_, ))
        self.robot_traj = []
        self.human_traj = []

        # stores the beliefs and cost history
        self.belief_hist = []
        self.cost_hist = []

        # goals
        self.xr_goal = np.zeros((2, ))
        self.xh_goal = np.zeros((2, ))

        # flags for data update
        self.flag_plan_updated = False
        self.flag_ctrl_updated = False
        self.flag_comm_updated = False

        # subscribers and publishers
        self.comm_sub = rospy.Subscriber("/planner/communication", Int32, self.comm_callback)
        self.ctrl_sub = rospy.Subscriber("/planner/cmd_vel", Twist, self.robot_ctrl_callback)
        self.plan_sub = rospy.Subscriber("/planner/full_plan", PlannedTrajectories, self.plan_callback)
        self.belief_cost_sub = rospy.Subscriber("/planner/belief_and_costs",
                                                Float64MultiArray, self.belief_cost_callback)

        self.robot_state_pub = rospy.Publisher("/amcl_pose", PoseWithCovarianceStamped, queue_size=1)
        self.robot_vel_pub = rospy.Publisher("/odom", Odometry, queue_size=1)
        self.human_state_pub = rospy.Publisher("/tracked_human", Float64MultiArray, queue_size=1)
        self.goal_pub = rospy.Publisher("/planner/set_goal", Float64MultiArray, queue_size=1)
        self.planner_pause_pub = rospy.Publisher("/planner/pause", Bool, queue_size=1)

    def load_data(self, path, test_id):
        # load initial states, human trajectory, goal states
        init_data = np.loadtxt(path + "/init.txt", delimiter=",")
        goal_data = np.loadtxt(path + "/goal.txt", delimiter=",")
        traj_data = np.loadtxt(path + "/test" + str(test_id) + ".txt", delimiter=",")

        self.xh_goal = goal_data[test_id, 0:2]
        self.xr_goal = goal_data[test_id, 2:4]
        self.xh0 = init_data[test_id, 0:self.nXh_]
        self.xr0 = init_data[test_id, self.nXh_:(self.nXh_+self.nXr_)]

        self.human_traj = traj_data[:, 0:self.nXh_]

        # FIXME: fix the intent for now
        self.robot_intent_ = 0  # human priority

    # main function to run
    def run_simulation(self):
        # total "simulation" time is length of pre-defined human trajectory
        Tsim = len(self.human_traj)

        # create subplots for each frame
        # 4 plots each row
        n_cols = 4
        n_rows = (Tsim - 1) / n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols)

        self.xh_ = self.xh0
        self.xr_ = self.xr0

        for t in range(Tsim):
            print "At time step t = ", t*self.dt_

            # publish the current human state
            self.publish_states()

            # tell the planner to start if t = 0
            # otherwise tell the planner to stop pausing
            if t == 0:
                goal_data = Float64MultiArray()
                for xr in self.xr_goal:
                    goal_data.data.append(xr)
                for xh in self.xh_goal:
                    goal_data.data.append(xh)

                # set intent data
                goal_data.data.append(self.robot_intent_)

                self.goal_pub.publish(goal_data)
            else:
                pause_data = Bool()
                pause_data.data = False

                self.planner_pause_pub.publish(pause_data)

            # wait for the planner to finish
            while not (self.flag_plan_updated and self.flag_ctrl_updated):
                if rospy.is_shutdown():
                    return
                rospy.sleep(0.01)

            # reset the flags
            self.flag_plan_updated = False
            self.flag_ctrl_updated = False

            # execute the control and update pose and vel of the robot
            print "executing control: ", self.robot_ctrl_
            self.xr_, self.ur_ = self.robot_dynamics(self.xr_, self.robot_ctrl_)

            # append to full trajectory
            self.robot_traj.append(self.xr_.copy())

            # update pose and vel of human
            self.xh_ = self.human_traj[t]

            # visualize the plan
            row = t / n_cols
            col = t % n_cols
            self.visualize_frame(axes[row][col], t+1)

        # visualize beliefs and partial costs
        self.visualize_belief_and_costs()

        # show visualization
        plt.show()

    # helper functions
    def publish_states(self):
        # robot state
        robot_state = PoseWithCovarianceStamped()
        robot_state.pose.pose.position.x = self.xr_[0]
        robot_state.pose.pose.position.y = self.xr_[1]
        robot_state.pose.pose.orientation.w = np.cos(self.xr_[2] * 0.5)
        robot_state.pose.pose.orientation.z = np.sin(self.xr_[2] * 0.5)

        self.robot_state_pub.publish(robot_state)

        # robot vel
        odom_data = Odometry()
        odom_data.twist.twist.linear.x = self.ur_[0]
        odom_data.twist.twist.angular.z = self.ur_[1]

        self.robot_vel_pub.publish(odom_data)

        # human state
        human_state = Float64MultiArray()
        for xh in self.xh_:
            human_state.data.append(xh)

        self.human_state_pub.publish(human_state)

    # visualize the "frame"
    def visualize_frame(self, ax, t):
        # plot previous trajectories
        robot_traj = np.asarray(self.robot_traj[:t]).reshape(t, self.nXr_)
        human_traj = np.asarray(self.human_traj[:t]).reshape(t, self.nXh_)

        ax.plot(robot_traj[:, 0], robot_traj[:, 1], "-o",
                color=(0.3, 0.3, 0.9), fillstyle="none", lw=1.5, label="robot_traj")
        ax.plot(human_traj[:, 0], human_traj[:, 1], "-o",
                color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")

        # plot the plan
        robot_plan = self.robot_traj_opt_.reshape(self.T_, self.nXr_)
        human_plan_hp = self.human_traj_hp_opt_.reshape(self.T_, self.nXh_)
        human_plan_rp = self.human_traj_rp_opt_.reshape(self.T_, self.nXh_)
        ax.plot(robot_plan[:, 0], robot_plan[:, 1], "-",
                color=(0.3, 0.3, 0.9, 0.5), lw=1.0, label="robot_plan")
        ax.plot(human_plan_hp[:, 0], human_plan_hp[:, 1], "-",
                color=(0.1, 0.1, 0.1, 0.5), lw=1.0, label="human_pred_hp")
        ax.plot(human_plan_rp[:, 0], human_plan_rp[:, 1], "--",
                color=(0.1, 0.1, 0.1, 0.5), lw=1.0, label="human_pred_rp")

        # plot the goals
        ax.plot(self.xr_goal[0], self.xr_goal[1], 'ob')
        ax.plot(self.xh_goal[0], self.xh_goal[1], 'ok')

        ax.axis("equal")
        ax.set_title("t = " + str(t))

    # visualize belief changes and partial costs
    def visualize_belief_and_costs(self):
        beliefs = np.asarray(self.belief_hist)
        costs = np.asarray(self.cost_hist)

        fig, ax = plt.subplots(2, 1)
        ax[0].plot(beliefs, '-ks', lw=1.5)
        ax[0].set_title("belief of human having priority")

        ax[1].plot(costs[:, 0], '-bs', lw=1.5, fillstyle="none", label="cost no communication")
        ax[1].plot(costs[:, 1], '--b^', lw=1.5, fillstyle="none", label="cost communication")
        ax[1].set_title("robot intent is" + str(self.robot_intent_))
        ax[1].legend()

        # plot for partial costs
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(costs[:, 2], '-bs', lw=1.5, fillstyle="none", label="cost hp")
        ax[0].plot(costs[:, 3], '--b^', lw=1.5, fillstyle="none", label="cost rp")
        ax[0].legend()
        ax[0].set_title("no communication")

        ax[1].plot(costs[:, 4], '-bs', lw=1.5, fillstyle="none", label="cost hp")
        ax[1].plot(costs[:, 5], '--b^', lw=1.5, fillstyle="none", label="cost rp")
        ax[1].legend()
        ax[1].set_title("with communication")

    # robot dynamics
    def robot_dynamics(self, x, u):
        # get parameters
        v_max = rospy.get_param("~robot_dynamics/v_max", [0.55, 3.0])
        a_max = rospy.get_param("~robot_dynamics/a_max", [1.0, 3.0])
        v_std = rospy.get_param("~robot_dynamics/v_std", [0.0, 0.0])

        v_max = np.asarray(v_max)
        a_max = np.asarray(a_max)
        v_std = np.asarray(v_std)

        # try to reach the commanded velocity
        u = np.clip(u, -v_max, v_max)
        u = np.clip(u, self.ur_ - a_max * self.dt_, self.ur_ + a_max * self.dt_)

        # sample velocity noise
        if v_std[0] > 0:
            dv = np.random.normal(0.0, v_std[0], 1)
        else:
            dv = 0.0

        if v_std[1] > 0:
            dom = np.random.normal(0.0, v_std[1], 1)
        else:
            dom = 0.0

        u += np.array([dv, dom])

        # update position
        th = x[2]
        th_new = th + u[1] * self.dt_

        if np.abs(u[1]) > 1e-3:
            R = u[0] / u[1]
            x[0] += R * (np.sin(th_new) - np.sin(th))
            x[1] -= R * (np.cos(th_new) - np.cos(th))
        else:
            x[0] += u[0] * np.cos(th) * self.dt_
            x[1] += u[0] * np.sin(th) * self.dt_

        x[2] = th_new

        return x, u

    # callbacks
    def comm_callback(self, comm_msg):
        self.acomm_ = comm_msg.data
        self.flag_comm_updated = True
        print "received communication: ", self.acomm_

    def robot_ctrl_callback(self, ctrl_msg):
        self.robot_ctrl_[0] = ctrl_msg.linear.x
        self.robot_ctrl_[1] = ctrl_msg.angular.z
        self.flag_ctrl_updated = True

    def plan_callback(self, plan_msg):
        self.robot_traj_opt_ = np.asarray(plan_msg.robot_traj_opt)
        self.human_traj_hp_opt_ = np.asarray(plan_msg.human_traj_hp_opt)
        self.human_traj_rp_opt_ = np.asarray(plan_msg.human_traj_rp_opt)
        self.flag_plan_updated = True

    def belief_cost_callback(self, msg):
        self.belief_hist.append(msg.data[0])
        self.cost_hist.append(msg.data[1:7])


if __name__ == "__main__":
    rospy.init_node("planner_simulator")

    simulator = PlannerSimulator()
    simulator.load_data("/home/yuhang/Documents/hri_log/test_data", 0)
    simulator.run_simulation()
