#include <cmath>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/LaserScan.h>
#include <boost/bind.hpp>



class RobotModel {
private:
    double x = 0.0, y = 0.0, alpha = 0.0; // Position and orientation
    double L = 1.5; // Wheelbase
    double f = 0.2; // offset
    double r = 0.1; // wheel radius

    double vx = 0.0;
    double vy = 0.0;
    sensor_msgs::LaserScan modified_msg;
    ros::Time last_update_time_;
    ros::Publisher odom_pub;
    ros::Subscriber joint_states_sub;
    ros::Subscriber laser_scan_sub;
    ros::Publisher modified_laser_scan_pub;

    tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;

    void updatePose(double v, double theta, ros::Time current_time) {
        
        double dt = (current_time - last_update_time_).toSec();
        if (dt>0){
            double R = (L / tan(theta)) + f;
            double omega = v / R;
            
            double deltaAlpha = omega * dt;
            alpha += deltaAlpha;
            
            alpha = atan2(sin(alpha), cos(alpha));
            
            double deltaX;
            double deltaY;


            if (fabs(theta) == 0) { // Check for straight movement
                deltaX = v * cos(alpha) * dt;
                deltaY = v * sin(alpha) * dt;
            } else { // For turning, adjust calculations to reflect arc movement
                deltaX = R * sin(alpha + deltaAlpha) - R * sin(alpha);
                deltaY = -R * cos(alpha + deltaAlpha) + R * cos(alpha);
            }

            vx = deltaX/dt;
            vy = deltaY/dt;

            x += deltaX;
            y += deltaY;
        
            publishOdometry(x, y, alpha, v, omega, current_time);
            last_update_time_ = current_time;
        }
    }

    void publishOdometry(double x, double y, double alpha, double v, double omega, ros::Time current_time) {
        
        sensor_msgs::LaserScan new_msg = this->modified_msg;
        new_msg.header.stamp = current_time;

        bool isEmpty = std::all_of(new_msg.ranges.begin(), new_msg.ranges.end(), [](float range) {
            return std::isinf(range);
        });

        if (!isEmpty) {
            // The message is not "empty", so we proceed to copy and modify it
                modified_laser_scan_pub.publish(new_msg);
            }
        
        
        nav_msgs::Odometry odom_msg;
        odom_msg.header.stamp = current_time;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "robot_frame";
        
        odom_msg.pose.pose.position.x = x;
        odom_msg.pose.pose.position.y = y;
        tf2::Quaternion q;
        q.setRPY(0, 0, alpha);
        odom_msg.pose.pose.orientation = tf2::toMsg(q);
        
        odom_msg.twist.twist.linear.x = vx;
        odom_msg.twist.twist.linear.y = vy;
        odom_msg.twist.twist.angular.z = omega;
        
        odom_pub.publish(odom_msg);
        
        transformStamped.header.stamp = current_time;
        transformStamped.header.frame_id = "odom";
        transformStamped.child_frame_id = "robot_frame";
        transformStamped.transform.translation.x = x;
        transformStamped.transform.translation.y = y;
        transformStamped.transform.translation.z = 0.0;
        transformStamped.transform.rotation = tf2::toMsg(q);

        br.sendTransform(transformStamped);
        
        
    }

public:
    RobotModel(ros::NodeHandle& nh) : last_update_time_(ros::Time::now()) {
        odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 1000);
        modified_laser_scan_pub = nh.advertise<sensor_msgs::LaserScan>("modified_laser_scan", 1000);
        laser_scan_sub = nh.subscribe<sensor_msgs::LaserScan>("/frankenstein/laser/scan", 1000, boost::bind(&RobotModel::laserScanCallback, this, _1));
        joint_states_sub = nh.subscribe<sensor_msgs::JointState>("/frankenstein/joint_states", 1000, &RobotModel::jointStatesCallback, this);
        
        
    }


    void laserScanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {
        // Create a new LaserScan message from the received one
        this->modified_msg = *msg;
    }

    void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        
        ROS_INFO("Received JointState message with %lu names, %lu velocities, and %lu positions", 
                msg->name.size(), msg->velocity.size(), msg->position.size());

        if (msg->name.size() < 2 || msg->velocity.size() < 1 || msg->position.size() < 2) {
            ROS_WARN("JointState message does not contain enough data for processing.");
            return;
        }

        // Additional checks to ensure safety
        if (msg->velocity.empty() || msg->position.size() < msg->name.size()) {
            ROS_ERROR("Unexpected JointState message structure.");
            return;
        }
        double wheel_angular_velocity = msg->velocity[0];
        double linear_velocity = wheel_angular_velocity * r;
        double steering_angle = msg->position[1];
        ros::Time current_time = msg->header.stamp;
        
        updatePose(linear_velocity, steering_angle, current_time);
    }

};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "robot_model");
    ros::NodeHandle nh;
    RobotModel robotModel(nh);

    ros::spin();

    return 0;
}
