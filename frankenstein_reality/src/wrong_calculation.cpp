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
#include <array>



class RobotModel {
private:
    double var_x = 0.1;
    double var_y = 0.1;
    double var_theta = 0.1;

    std::array<std::array<double, 3>, 3> Pk = {{
        {var_x, 0, 0},  // Covariance in x
        {0, var_y, 0},  // Covariance in y
        {0, 0, var_theta}  // Covariance in theta
    }};
    double x = 0.0, y = 0.0, alpha = 0.0; // Position and orientation
    double L = 1.33595; // Wheelbase
    double f = 0.267; // offset
    double r = 0.2322/2; // wheel radius

    double vx = 0.0;
    double vy = 0.0;
    sensor_msgs::LaserScan modified_msg;
    ros::Time last_update_time_;
    ros::Publisher odom_pub;
    ros::Subscriber joint_states_sub;

    tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;

    double previous_position = 0.0; 
    ros::Time previous_time; 

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
        odom_msg.pose.covariance = {Pk[0][0], Pk[0][1], 0, 0, 0, Pk[0][2],
                            Pk[1][0], Pk[1][1], 0, 0, 0, Pk[1][2],
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            Pk[2][0], Pk[2][1], 0, 0, 0, Pk[2][2]};
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
        joint_states_sub = nh.subscribe<sensor_msgs::JointState>("/frankenstein/joint_states", 1000, &RobotModel::jointStatesCallback, this);
        
        
    }

    void jointStatesCallback(const sensor_msgs::JointState::ConstPtr& msg) {
        
        if (msg->position.size() < 2) {
            ROS_WARN("Not enough position data.");
            return;
        }
        
        ros::Time current_time = msg->header.stamp;
        if (previous_time.toSec() > 0) { // Ensure it's not the first measurement
            double delta_time = (current_time - previous_time).toSec();
            if (delta_time <= 0) {
                ROS_WARN("Time delta is non-positive.");
                return;
            }

            // Calculate the change in position considering the wrap-around
            double current_position_deg = msg->position[0]; // Assuming this is the wheel position in degrees
            double delta_position_deg = current_position_deg - previous_position;

            // Adjust for wrap-around
            if (delta_position_deg > 180) {
                delta_position_deg -= 360;
            } else if (delta_position_deg < -180) {
                delta_position_deg += 360;
            }

            // Convert delta position from degrees to radians for angular velocity calculation
            double delta_position_rad = delta_position_deg * (M_PI / 180.0);

            // Estimate angular velocity of the wheel (in radians per second)
            double wheel_angular_velocity = delta_position_rad / delta_time;

            // Update for the next callback
            previous_position = current_position_deg;
            previous_time = current_time;

            // Use wheel_angular_velocity as needed
            // Note: If you need to calculate linear velocity from angular velocity, you would multiply by the wheel radius (in meters)
            double linear_velocity = wheel_angular_velocity * r;
            double steering_angle = msg->position[1]* (M_PI / 180.0); 
            updatePose(linear_velocity, steering_angle, current_time);
        } else {
            // First call, just update previous values
            previous_position = msg->position[0];
            previous_time = current_time;
        }
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
