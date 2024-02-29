#!/usr/bin/env python3
import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import time
time.sleep(1)
# Global variable to store the time of the last transform broadcast
last_broadcast_time = rospy.Time(0)

def handle_robot_pose(msg):
    global last_broadcast_time
    current_time = rospy.Time.now()

    # Check if enough time has passed
    if (current_time - last_broadcast_time).to_sec() < 0.1:  # 10 Hz
        return  # Skip this message if not enough time has passed

    br = tf2_ros.TransformBroadcaster()
    t = TransformStamped()

    t.header.stamp = msg.header.stamp
    t.header.frame_id = "odom"
    t.child_frame_id = "robot_frame"
    t.transform.translation.x = msg.pose.pose.position.x
    t.transform.translation.y = msg.pose.pose.position.y
    t.transform.translation.z = 0.0

    t.transform.rotation.x = msg.pose.pose.orientation.x
    t.transform.rotation.y = msg.pose.pose.orientation.y
    t.transform.rotation.z = msg.pose.pose.orientation.z
    t.transform.rotation.w = msg.pose.pose.orientation.w

    br.sendTransform(t)
    last_broadcast_time = current_time  # Update the time of the last broadcast

if __name__ == '__main__':
    rospy.init_node('robot_tf_broadcaster')
    rospy.Subscriber('/odom', Odometry, handle_robot_pose)
    rospy.spin()
