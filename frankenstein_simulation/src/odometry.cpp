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
#include <visualization_msgs/Marker.h>



class RobotModel {
private:
    double var_v = 0.000001;
    double var_steer = 0.000001;
    std::array<std::array<double, 3>, 3> P = {{{0}}};

    // Process noise covariance matrix (Q)
    std::array<std::array<double, 2>, 2> Q = {{
        {var_v, 0},
        {0, var_steer}
    }};
    double x = 0.0, y = 0.0, alpha = 0.0; // Position and orientation
    double L = 1.5; // Wheelbase
    double f = 0.2; // offset
    double r = 0.1; // wheel radius

    double vx = 0.0;
    double vy = 0.0;
    sensor_msgs::LaserScan modified_msg;
    ros::Time last_update_time_;
    ros::Publisher odom_pub;
    ros::Publisher marker_pub;
    ros::Subscriber joint_states_sub;

    tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped transformStamped;

    void updatePose(double v, double theta, ros::Time current_time) {
        
        double dt = (current_time - last_update_time_).toSec();
        if (dt>0){
            double omega = (v*sin(theta)) / L;
            double Vtcp = v*cos(theta) - (omega * f);

            alpha += omega*dt;
            double IncDist = Vtcp * dt ;
            double deltaX = IncDist*cos(alpha);
            double deltaY = IncDist*sin(alpha);

            vx = deltaX/dt;
            vy = deltaY/dt;

            x += deltaX;
            y += deltaY;

            // Approximate Jacobian of motion model w.r.t. state variables (F_x)
            std::array<std::array<double, 3>, 3> F_x = {{
                {1, 0, -deltaY},
                {0, 1, deltaX},
                {0, 0, 1}
            }};

            // Approximate Jacobian of motion model w.r.t. control inputs (F_u)
            // Note: These are simplified and should be derived based on your specific model
            std::array<std::array<double, 2>, 3> F_u = {{
                {cos(alpha) * dt, -IncDist * sin(alpha)},
                {sin(alpha) * dt, IncDist * cos(alpha)},
                {0, dt / L}
            }};

            

            // Update the covariance matrix P
            std::array<std::array<double, 3>, 3> tempP = P; // Copy of the original P for calculations

            // First term: F_x * P * F_x^T
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    P[i][j] = 0;
                    for (int k = 0; k < 3; ++k) {
                        for (int l = 0; l < 3; ++l) {
                            P[i][j] += F_x[i][k] * tempP[k][l] * F_x[j][l];
                        }
                    }
                }
            }

            // Second term: F_u * Q * F_u^T
            std::array<std::array<double, 3>, 3> FuQT = {{{0}}};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    for (int k = 0; k < 2; ++k) {
                        FuQT[i][j] += F_u[i][k] * Q[k][k] * F_u[j][k]; 
                    }
                    P[i][j] += FuQT[i][j]; 
                }
            }
        
            publishOdometry(x, y, alpha, v, omega, current_time);
            publishEllipse(x, y, P, current_time);
            last_update_time_ = current_time;
        }
    }

    void publishEllipse(double x, double y, const std::array<std::array<double, 3>, 3>& P, ros::Time current_time) {
        visualization_msgs::Marker ellipse;
        ellipse.header.frame_id = "odom";
        ellipse.header.stamp = current_time;
        ellipse.ns = "odom";
        ellipse.id = 0;
        ellipse.type = visualization_msgs::Marker::CYLINDER;
        ellipse.action = visualization_msgs::Marker::ADD;

        ellipse.pose.position.x = x;
        ellipse.pose.position.y = y;
        ellipse.pose.position.z = 0;

        // Calculate ellipse orientation and scale based on covariance matrix P
        double sxx = P[0][0];
        double syy = P[1][1];
        double sxy = P[0][1];

        // Eigenvalues of the covariance matrix represent the squared length of the ellipse axes
        double a = sxx + syy;
        double b = sqrt((sxx - syy) * (sxx - syy) + 4 * sxy * sxy);
        double lambda1 = (a + b) / 2; // Major axis variance
        double lambda2 = (a - b) / 2; // Minor axis variance

        ellipse.scale.x = sqrt(lambda1) * 2; // Convert variance to standard deviation and double for diameter
        ellipse.scale.y = sqrt(lambda2) * 2; // Convert variance to standard deviation and double for diameter
        ellipse.scale.z = 0.01; // Thin cylinder to represent an ellipse on the ground plane

        // Orientation of the ellipse (derived from the eigenvectors)
        double angle = 0.5 * atan2(2 * sxy, sxx - syy);
        tf2::Quaternion q_ellipse;
        q_ellipse.setRPY(0, 0, angle);
        ellipse.pose.orientation = tf2::toMsg(q_ellipse);

        ellipse.color.a = 0.3; // Transparency
        ellipse.color.r = 0.0;
        ellipse.color.g = 1.0;
        ellipse.color.b = 0.0;

        // Publish the ellipse marker
        marker_pub.publish(ellipse);
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
        odom_msg.pose.covariance = {P[0][0], P[0][1], 0, 0, 0, P[0][2],
                            P[1][0], P[1][1], 0, 0, 0, P[1][2],
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            P[2][0], P[2][1], 0, 0, 0, P[2][2]};
        odom_pub.publish(odom_msg);
        
        // transformStamped.header.stamp = current_time;
        // transformStamped.header.frame_id = "odom";
        // transformStamped.child_frame_id = "robot_frame";
        // transformStamped.transform.translation.x = x;
        // transformStamped.transform.translation.y = y;
        // transformStamped.transform.translation.z = 0.0;
        // transformStamped.transform.rotation = tf2::toMsg(q);

        // br.sendTransform(transformStamped);
        
        
    }

public:
    RobotModel(ros::NodeHandle& nh) : last_update_time_(ros::Time::now()) {
        odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 1000);
        joint_states_sub = nh.subscribe<sensor_msgs::JointState>("/frankenstein/joint_states", 1000, &RobotModel::jointStatesCallback, this);
        marker_pub = nh.advertise<visualization_msgs::Marker>("elipse_odometry", 1000);
        
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
