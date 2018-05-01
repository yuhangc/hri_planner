#!/usr/bin/env python

import numpy as np

import rospy
from geometry_msgs.msg import Twist


class VelocitySmoother:
    def __init__(self):
        # parameters
        self.time_out_high = rospy.get_param("~time_out_high", 1.5)
        self.time_out_low = rospy.get_param("~time_out_low", 0.5)
        self.decay_rate = rospy.get_param("~decay_rate", 0.9)

        self.cmd_vel = Twist()
        self.t_cmd_last = 0.0

        self.cmd_vel_sub = rospy.Subscriber("cmd_vel",
                                            Twist, self.cmd_vel_callback)
        self.cmd_vel_pub = rospy.Publisher("cmd_vel_smooth", Twist, queue_size=1)

    def cmd_vel_callback(self, msg):
        self.cmd_vel = msg
        self.t_cmd_last = rospy.get_time()

    def run(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            rate.sleep()

            # check if time out
            t = rospy.get_time() - self.t_cmd_last

            if t > self.time_out_low:
                if t > self.time_out_high:
                    self.cmd_vel = Twist()
                else:
                    self.cmd_vel.linear.x *= self.decay_rate
                    self.cmd_vel.angular.z *= self.decay_rate

            self.cmd_vel_pub.publish(self.cmd_vel)

if __name__ == "__main__":
    rospy.init_node("planner_velocity_smoother")

    smoother = VelocitySmoother()
    smoother.run()
