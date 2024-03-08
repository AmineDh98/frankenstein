#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <tf/tf.h>

class DataParser {
public:
    DataParser(const std::string& file_path)
    : file_path_(file_path) {
        joint_pub_ = nh_.advertise<sensor_msgs::JointState>("joint_states", 10);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("/ground_truth", 10);
        laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>("lidar_scan", 10);
    }

    void parseAndPublish() {
    std::ifstream file(file_path_);
    if (!file.is_open()) {
        ROS_ERROR("Failed to open file.");
        return;
    }

    std::string line1, line2;
    while (ros::ok()) {
        // Read the first line for joint states and odometry
        if (!std::getline(file, line1)) break; // Break if there's no more data

        sensor_msgs::JointState joint_state;
        nav_msgs::Odometry odom;

        // Parse the first line
        parseJointAndOdometryData(line1, joint_state, odom);

        // Immediately read the next line for LIDAR data, assuming every valid joint/odom line is followed by a LIDAR line
        if (!std::getline(file, line2)) {
            ROS_ERROR("LIDAR data line expected after joint/odom line, but file ended prematurely.");
            break; // This error handling is simplistic; adjust as necessary for your application's needs
        }

        sensor_msgs::LaserScan laser_scan;
        // Parse the second line
        parseLidarData(line2, laser_scan);

        // Now publish the data
        joint_pub_.publish(joint_state);
        odom_pub_.publish(odom);
        laser_pub_.publish(laser_scan);

        ros::spinOnce();
        ros::Duration(0.01).sleep(); // Adjust sleep as necessary
    }
}

private:
    ros::NodeHandle nh_;
    ros::Publisher joint_pub_, odom_pub_, laser_pub_;
    std::string file_path_;

    void parseJointAndOdometryData(const std::string& line, sensor_msgs::JointState& joint_state, nav_msgs::Odometry& odom) {
        std::istringstream stream(line);
        std::string part;
        std::vector<std::string> parts;

        // Split line into parts
        while (std::getline(stream, part, ',')) {
            parts.push_back(part);
        }

        if (parts.size() < 7) {
            ROS_ERROR("Joint/Odometry data line does not contain enough parts.");
            return;
        }

        // Assuming the timestamp is at parts[0] and can be ignored for this example
        joint_state.header.stamp = ros::Time::now();
        joint_state.name = {"front_right_wheel_joint", "front_right_steering_joint"};
        joint_state.position = {std::stod(parts[1]), std::stod(parts[3])}; // Joint positions

        odom.header.stamp = joint_state.header.stamp;
        odom.pose.pose.position.x = std::stod(parts[4]);
        odom.pose.pose.position.y = std::stod(parts[5]);
        // Assuming the orientation can be approximated by converting theta to quaternion (simplistic approach)
        tf::Quaternion q = tf::createQuaternionFromYaw(std::stod(parts[6]));
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        odom.pose.pose.orientation.w = q.w();
    }

    

    void parseLidarData(const std::string& line, sensor_msgs::LaserScan& laser_scan) {
        std::istringstream stream(line);
        std::string part;
        std::vector<float> data;

        while (std::getline(stream, part, ',')) {
            try {
                float value = std::stof(part);
                data.push_back(value);
            } catch (const std::exception& e) {
                ROS_ERROR("Exception caught converting string to float: %s", e.what());
                return;
            }
        }

        if (data.size() % 2 != 0) {
            ROS_ERROR("LIDAR data line does not contain an even number of elements.");
            return;
        }

        // Assuming the first pair represents angle_min and the last pair represents angle_max
        laser_scan.angle_min = data[1];
        laser_scan.angle_max = data[data.size() - 1];
        laser_scan.angle_increment = (laser_scan.angle_max - laser_scan.angle_min) / ((data.size() / 2) - 1);
        laser_scan.time_increment = 0;
        laser_scan.range_min = 0.1; 
        laser_scan.range_max = 40; 

        laser_scan.header.stamp = ros::Time::now();

        // Assuming even indices are distances and odd indices are angles
        for (size_t i = 0; i < data.size(); i += 2) {
            laser_scan.ranges.push_back(data[i]);
        }
    }
   
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_parser");
    
    std::string file_path = "/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data/lidar_data3.txt";
    
    DataParser parser(file_path);
    
    parser.parseAndPublish();

    ros::spin();
    return 0;
}
