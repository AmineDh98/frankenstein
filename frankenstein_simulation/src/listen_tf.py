import rospy
import tf2_ros

def callback(trans):
    print("Received a transform from {} to {} at time {}.".format(
        trans.header.frame_id, trans.child_frame_id, trans.header.stamp))

if __name__ == '__main__':
    rospy.init_node('tf_listener')

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('odom', 'robot_frame', rospy.Time())
            callback(trans)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            continue

        rate.sleep()
