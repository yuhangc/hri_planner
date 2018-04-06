#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import Int32
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from people_msgs.msg import People
from people_msgs.msg import Person

from hri_planner.msg import PlannedTrajectories


def wrap_to_pi(ang):
    while ang >= np.pi:
        ang -= 2.0 * np.pi
    while ang < -np.pi:
        ang += 2.0 * np.pi

    return ang


class PlannerSimulator(object):
    def __init__(self):
        # dimensions
        self.T_ = rospy.get_param("~dimension/T", 6)
        self.nXh_ = rospy.get_param("~dimension/nXh", 4)
        self.nUh_ = rospy.get_param("~dimension/nUh", 2)
        self.nXr_ = rospy.get_param("~dimension/nXr", 3)
        self.nUr_ = rospy.get_param("~dimension/nUr", 2)
        self.dt_ = rospy.get_param("~dimension/dt", 0.5)

        # thresholds to simulate detection range
        self.dist_th_detection_ = rospy.get_param("~sensing/dist_th", 4.0)
        self.ang_th_detection_ = rospy.get_param("~sensing/ang_th", 0.6*np.pi)

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
        # self.human_state_pub = rospy.Publisher("/tracked_human", Float64MultiArray, queue_size=1)
        self.human_tracking_pub = rospy.Publisher("/people", People, queue_size=1)
        self.goal_pub = rospy.Publisher("/planner/set_goal", Float64MultiArray, queue_size=1)
        self.planner_ctrl_pub = rospy.Publisher("/planner/ctrl", String, queue_size=1)

    def load_data(self, path, test_id):
        # load initial states, human trajectory, goal states
        init_data = np.loadtxt(path + "/init.txt", delimiter=",")
        goal_data = np.loadtxt(path + "/goal.txt", delimiter=",")
        traj_data = np.loadtxt(path + "/test" + str(test_id) + ".txt", delimiter=",")

        self.xh_goal = goal_data[test_id, 0:2]
        self.xr_goal = goal_data[test_id, 2:4]
        self.xh0 = init_data[test_id, 0:self.nXh_]
        self.xr0 = init_data[test_id, self.nXh_:(self.nXh_+self.nXr_)]

        self.human_traj = traj_data

    def save_data(self, path, test_id):
        traj_data = np.hstack((self.human_traj, self.robot_traj))

        np.savetxt(path + "/block" + str(test_id) + ".txt", traj_data, delimiter=',')

    # main function to run
    def run_simulation(self, robot_intent):
        # total "simulation" time is length of pre-defined human trajectory
        Tsim = len(self.human_traj)

        # create subplots for each frame
        # 4 plots each row
        n_cols = 4
        n_rows = (Tsim - 1) / n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols)

        self.xh_ = self.xh0
        self.xr_ = self.xr0

        # publish goal first
        goal_data = Float64MultiArray()
        for xr in self.xr_goal:
            goal_data.data.append(xr)
        for xh in self.xh_goal:
            goal_data.data.append(xh)
        for xh in self.xh0:
            goal_data.data.append(xh)

        # set intent data
        goal_data.data.append(robot_intent)

        self.goal_pub.publish(goal_data)

        for t in range(Tsim):
            print "At time step t = ", t*self.dt_

            # publish the current human state
            self.publish_states()

            # tell the planner to start if t = 0
            # otherwise tell the planner to stop pausing
            if t == 0:
                ctrl_data = String()
                ctrl_data.data = "start"
                self.planner_ctrl_pub.publish(ctrl_data)
            else:
                ctrl_data = String()
                ctrl_data.data = "resume"
                self.planner_ctrl_pub.publish(ctrl_data)

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
            self.robot_traj.append(np.hstack((self.xr_.copy(), self.ur_.copy())))

            # update pose and vel of human
            self.xh_ = self.human_traj[t, 0:self.nXh_]

            # visualize the plan
            row = t / n_cols
            col = t % n_cols
            self.visualize_frame(axes[row][col], t+1)

        # tell the planner to stop
        ctrl_data = String()
        ctrl_data.data = "stop"
        self.planner_ctrl_pub.publish(ctrl_data)

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
        # check if human is within detection range
        x_diff = self.xh_[0:2] - self.xr_[0:2]
        if np.linalg.norm(x_diff) > self.dist_th_detection_:
            return

        th_rel = wrap_to_pi(np.arctan2(x_diff[1], x_diff[0]) - self.xr_[2])
        if np.abs(th_rel) > self.ang_th_detection_:
            return

        people_states = People()
        person_state = Person()

        person_state.position.x = self.xh_[0]
        person_state.position.y = self.xh_[1]
        person_state.velocity.x = self.xh_[2]
        person_state.velocity.y = self.xh_[3]

        people_states.people.append(person_state)

        self.human_tracking_pub.publish(people_states)

    # visualize the "frame"
    def visualize_frame(self, ax, t):
        # plot previous trajectories
        robot_traj = np.asarray(self.robot_traj)[0:t, 0:self.nXr_].reshape(t, self.nXr_)
        human_traj = np.asarray(self.human_traj)[0:t, 0:self.nXh_].reshape(t, self.nXh_)

        ax.plot(robot_traj[:, 0], robot_traj[:, 1], "-o",
                color=(0.3, 0.3, 0.9), fillstyle="none", lw=1.5, label="robot_traj")
        ax.plot(human_traj[:, 0], human_traj[:, 1], "-o",
                color=(0.1, 0.1, 0.1), fillstyle="none", lw=1.5, label="human_traj")

        # plot the plan
        robot_plan = self.robot_traj_opt_.reshape(self.T_, self.nXr_)
        ax.plot(robot_plan[:, 0], robot_plan[:, 1], "-",
                color=(0.3, 0.3, 0.9, 0.5), lw=1.0, label="robot_plan")

        if self.human_traj_hp_opt_.size > 0:
            human_plan_hp = self.human_traj_hp_opt_.reshape(self.T_, self.nXh_)
            ax.plot(human_plan_hp[:, 0], human_plan_hp[:, 1], "-",
                    color=(0.1, 0.1, 0.1, 0.5), lw=1.0, label="human_pred_hp")

        if self.human_traj_rp_opt_.size > 0:
            human_plan_rp = self.human_traj_rp_opt_.reshape(self.T_, self.nXh_)
            ax.plot(human_plan_rp[:, 0], human_plan_rp[:, 1], "--",
                    color=(0.1, 0.1, 0.1, 0.5), lw=1.0, label="human_pred_rp")

        # plot the goals
        ax.plot(self.xr_goal[0], self.xr_goal[1], 'ob')
        ax.plot(self.xh_goal[0], self.xh_goal[1], 'ok')

        ax.axis("equal")
        ax.set_title("t = " + str(t))

    # visualize belief changes and partial costs
    def visualize_belief_and_costs(self):
        if not self.belief_hist:
            return

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
        v_max = rospy.get_param("~robot_dynamics/v_max", [0.50, 3.0])
        a_max = rospy.get_param("~robot_dynamics/a_max", [0.5, 2.0])
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
    simulator.run_simulation(0)
    simulator.save_data("/home/yuhang/Documents/hri_log/test_data", 0)
