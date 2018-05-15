#!/usr/bin/env python

import numpy as np

import rospy
from std_msgs.msg import Bool
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray

import actionlib
import move_base_msgs.msg
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped

from hri_planner.msg import PlannedTrajectories

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


class PlannerDataLogger(object):
    def __init__(self, save_path):
        # path to save data
        self.save_path = save_path

        # dimensions
        self.T = rospy.get_param("~dimension/T", 6)
        self.nXr = rospy.get_param("~dimension/nXr", 3)
        self.nXh = rospy.get_param("~dimension/nXh", 4)

        # information to save
        self.t_plan = []
        self.xr_hist = []
        self.xh_hist = []
        self.xr_hist_pred = []
        self.xh_hist_pred = []
        self.robot_traj_opt = []
        self.human_traj_hp_opt = []
        self.human_traj_rp_opt = []

        self.tracking_status_hist = []

        self.t_belief = []
        self.belief_hist = []
        self.cost_hist = []

        # log the communication for the entire experiment
        self.comm_logger = open(self.save_path + "/communication.txt", "w")
        self.comm_hist = []

        # set an initial trial number
        self.trial = 0

        # subscribers
        self.plan_sub = rospy.Subscriber("/planner/full_plan", PlannedTrajectories, self.plan_callback)
        self.robot_human_state_sub = rospy.Subscriber("/planner/robot_human_state",
                                                      Float64MultiArray, self.robot_human_state_callback)
        self.belief_cost_sub = rospy.Subscriber("/planner/belief_and_costs",
                                                Float64MultiArray, self.belief_cost_callback)

        # time start
        self.t_start = rospy.get_time()

    def reset(self, trial=-1):
        self.t_plan = []
        self.xr_hist = []
        self.xh_hist = []
        self.xr_hist_pred = []
        self.xh_hist_pred = []
        self.robot_traj_opt = []
        self.human_traj_hp_opt = []
        self.human_traj_rp_opt = []

        self.tracking_status_hist = []

        self.t_belief = []
        self.belief_hist = []
        self.cost_hist = []

        self.comm_hist = []

        if trial >= 0:
            self.trial = trial
        else:
            self.trial += 1

        self.t_start = rospy.get_time()

    def save_data(self):
        print "starting to save data for trial ", self.trial, "..."
        # time stamps
        np.savetxt(self.save_path + "/tstamp_plan" + str(self.trial) + ".txt",
                   np.asarray(self.t_plan).transpose(), delimiter=',')
        np.savetxt(self.save_path + "/tstamp_belief" + str(self.trial) + ".txt",
                   np.asarray(self.t_belief).transpose(), delimiter=',')

        # actual and predicted trajectories
        xr_hist = np.hstack((np.asarray(self.xr_hist), np.asarray(self.xr_hist_pred)))
        xh_hist = np.hstack((np.asarray(self.xh_hist), np.asarray(self.xh_hist_pred)))
        np.savetxt(self.save_path + "/robot_traj" + str(self.trial) + ".txt",
                   xr_hist, delimiter=',')
        np.savetxt(self.save_path + "/human_traj" + str(self.trial) + ".txt",
                   xh_hist, delimiter=',')

        # tracking status
        np.savetxt(self.save_path + "/tracking_state" + str(self.trial) + ".txt",
                   np.asarray(self.tracking_status_hist), delimiter=',')

        # planned trajectories
        np.savetxt(self.save_path + "/robot_plan" + str(self.trial) + ".txt",
                   np.asarray(self.robot_traj_opt), delimiter=',')
        np.savetxt(self.save_path + "/human_pred_hp" + str(self.trial) + ".txt",
                   np.asarray(self.human_traj_hp_opt), delimiter=',')
        np.savetxt(self.save_path + "/human_pred_rp" + str(self.trial) + ".txt",
                   np.asarray(self.human_traj_rp_opt), delimiter=',')

        # belief/costs
        np.savetxt(self.save_path + "/belief_hist" + str(self.trial) + ".txt",
                   np.asarray(self.belief_hist), delimiter=',')
        np.savetxt(self.save_path + "/cost_hist" + str(self.trial) + ".txt",
                   np.asarray(self.cost_hist), delimiter=',')

        # communication
        if self.comm_hist:
            self.comm_logger.write(", ".join(self.comm_hist) + "\n")
        else:
            self.comm_logger.write("No communication\n")

    def log_comm(self, acomm):
        self.comm_hist.append(acomm)

    def plan_callback(self, plan_msg):
        self.t_plan.append(rospy.get_time() - self.t_start)

        self.xr_hist_pred.append(np.asarray(plan_msg.xr_init))
        self.robot_traj_opt.append(np.asarray(plan_msg.robot_traj_opt))

        if plan_msg.xh_init:
            self.xh_hist_pred.append(np.asarray(plan_msg.xh_init))
            self.human_traj_hp_opt.append(np.asarray(plan_msg.human_traj_hp_opt))
            self.human_traj_rp_opt.append(np.asarray(plan_msg.human_traj_rp_opt))

            self.tracking_status_hist.append(plan_msg.tracking_lost)
        else:
            # set to zero if human tracking/prediction is missing
            self.xh_hist_pred.append(np.zeros((self.nXh, )))
            self.human_traj_hp_opt.append(np.zeros((self.T * self.nXh, )))
            self.human_traj_rp_opt.append(np.zeros((self.T * self.nXh, )))

            self.tracking_status_hist.append(0)

    def robot_human_state_callback(self, state_msg):
        state_arr = np.asarray(state_msg.data)
        self.xr_hist.append(state_arr[0:3])
        self.xh_hist.append(state_arr[3:7])

    def belief_cost_callback(self, msg):
        self.t_belief.append(rospy.get_time() - self.t_start)

        self.belief_hist.append(msg.data[0])
        self.cost_hist.append(msg.data[1:7])


class ExperimentManager(object):
    def __init__(self):
        self.comm_sub = rospy.Subscriber("/planner/communication", String, self.comm_callback)

        # load goals
        protocol_file = rospy.get_param("~protocol_file", "../../resources/exp_protocols/protocol.txt")

        self.acomm = ""
        self.flag_comm_updated = False

        self.flag_goal_reached = True

        # create a planner data logger
        self.save_path = rospy.get_param("~save_path", "exp_data")
        self.logger = PlannerDataLogger(self.save_path)

        self.xr_goal = []
        self.xh_goal = []
        self.xh_init = []
        self.intent = []

        self.load_goals(protocol_file)

        # subscribers and publishers
        self.goal_reach_sub = rospy.Subscriber("controller/goal_reached", Bool, self.goal_reached_callback)
        self.comm_sub = rospy.Subscriber("/planner/communication", String, self.comm_callback)

        self.goal_pub = rospy.Publisher("/planner/set_goal", Float64MultiArray, queue_size=1)
        self.planner_ctrl_pub = rospy.Publisher("/planner/ctrl", String, queue_size=1)
        self.tracking_test_pub = rospy.Publisher("/test_human_tracking_start", Bool, queue_size=1)
        self.haptic_ctrl_pub = rospy.Publisher("/haptic_control", String, queue_size=1)

    def load_goals(self, protocol_file):
        proto_data = np.loadtxt(protocol_file, delimiter=',')

        # parse the protocol file
        self.xr_goal = proto_data[:, 1:4]
        self.xh_goal = proto_data[:, 4:6]
        self.xh_init = proto_data[:, 6:8]
        self.intent = proto_data[:, 8]

        # save protocol file to save path
        np.savetxt(self.save_path + "/protocol.txt", proto_data, delimiter=', ', fmt="%.3f")

    def run(self, trial_start=0):
        rate = rospy.Rate(20)

        print "Please press any key to start:"
        getchar()

        trial = trial_start
        while not rospy.is_shutdown():
            rate.sleep()

            # publish new goal if goal reached
            if self.flag_goal_reached:
                # save data if applicable
                if trial > trial_start:
                    self.logger.save_data()

                # reset logger
                self.logger.reset(trial)

                if trial >= len(self.xr_goal):
                    rospy.loginfo("trials ended!")
                    break

                self.flag_goal_reached = False
                self.publish_goal(trial)
                trial += 1

                rate.sleep()

                print "Please press 's' to start:"
                while getchar() != 's':
                    rate.sleep()
                    print "Please press 's' to start:"

                # tell planner to start
                ctrl_data = String()
                ctrl_data.data = "start"
                self.planner_ctrl_pub.publish(ctrl_data)

                # tell tracking test to start
                tracking_start = Bool()
                tracking_start.data = True
                self.tracking_test_pub.publish(tracking_start)

                rospy.loginfo("started!")

    def run_baseline(self, trial_start=0):
        rate = rospy.Rate(20)

        # create an actionlib client
        rospy.loginfo("Creating human aware navigation client.")
        hri_nav_client = actionlib.SimpleActionClient(
            'human_aware_navigation',
            move_base_msgs.msg.MoveBaseAction
        )
        hri_nav_client.wait_for_server()
        rospy.loginfo("...done")

        print "Please press any key to start:"
        getchar()

        trial = trial_start
        seq = 0
        while not rospy.is_shutdown():
            rate.sleep()

            print "Please press 's' to start:"
            while getchar() != 's':
                rate.sleep()
                print "Please press 's' to start:"

            # send the haptic signal
            haptic_msg = String()
            if self.intent[trial] == 0:
                haptic_msg.data = "Attract"
            else:
                haptic_msg.data = "Repel"
            self.haptic_ctrl_pub.publish(haptic_msg)

            # set and send a navigation goal
            goal = move_base_msgs.msg.MoveBaseGoal()
            goal.target_pose.pose.position.x = self.xr_goal[trial][0]
            goal.target_pose.pose.position.y = self.xr_goal[trial][1]
            th = self.xr_goal[trial][2]
            goal.target_pose.pose.orientation.z = np.sin(th * 0.5)
            goal.target_pose.pose.orientation.w = np.cos(th * 0.5)
            goal.target_pose.header.frame_id = "map"

            seq += 1
            goal.target_pose.header.seq = seq
            goal.target_pose.header.stamp = rospy.get_rostime()

            hri_nav_client.send_goal(goal)

            # wait for result
            hri_nav_client.wait_for_result()

            # get result
            res = hri_nav_client.get_result()
            state = hri_nav_client.get_state()

            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo("Goal reached!")

                trial += 1
                flag_goal_reached = False
            else:
                # cancel and resend goal
                hri_nav_client.cancel_all_goals()

            if trial >= len(self.xr_goal):
                rospy.loginfo("trials ended!")
                break

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

        if self.goal_pub.get_num_connections() > 0:
            self.goal_pub.publish(goal_data)
            print "goal data sent!"
        else:
            raise RuntimeError("no subscriber!!!!!!!!!!!!!")

    def comm_callback(self, comm_msg):
        if comm_msg.data == "Attract":
            self.acomm = "HumanPriority"
        else:
            self.acomm = "RobotPriority"

        self.flag_comm_updated = True
        print "received communication: ", self.acomm

        self.logger.log_comm(self.acomm)

    def goal_reached_callback(self, msg):
        self.flag_goal_reached = msg.data
        print "Goal reached: ", msg.data


if __name__ == "__main__":
    rospy.init_node("human_aware_navigation_goal_publisher")

    # sleep after object creation to ensure that publishers and subscribers get connected
    exp_manager = ExperimentManager()
    rospy.sleep(0.5)

    # run differently based on mode
    run_baseline_planner = rospy.get_param("~run_baseline_planner", False)

    if run_baseline_planner:
        exp_manager.run_baseline(0)
    else:
        exp_manager.run(0)
