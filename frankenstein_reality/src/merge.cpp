#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <nav_msgs/Odometry.h>
#include <boost/asio.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <functional>
#include <boost/asio.hpp>
#include <cstring>
#include <cmath>
#include <sensor_msgs/LaserScan.h>
#include <thread>


using boost::asio::ip::udp;

class DataParser {
public:
    DataParser(boost::asio::io_service& io_service, const std::string& host, const std::string& port)
    : socket_(io_service) {
        // Construct endpoint from host and port
        udp::endpoint local_endpoint(udp::v4(), std::stoi(port));

        // Open the socket
        socket_.open(udp::v4());

        // Set SO_REUSEADDR option before binding
        boost::asio::socket_base::reuse_address option(true);
        socket_.set_option(option);

        // Now bind the socket
        socket_.bind(local_endpoint);

        startReceive();
        joint_pub_ = nh_.advertise<sensor_msgs::JointState>("/frankenstein/joint_states", 10);
        gt_pub_ = nh_.advertise<nav_msgs::Odometry>("/ground_truth", 10);
        lidar_pub_=nh_.advertise<sensor_msgs::LaserScan>("/scan", 50);
        lidar_sub_ = nh_.subscribe("/fast_scan", 50, &DataParser::scanCallback, this);

    }

    void startReceive() {
        socket_.async_receive_from(
            boost::asio::buffer(recv_buffer_), remote_endpoint_,
            [this](const boost::system::error_code& error, std::size_t bytes_transferred) {
                if (!error) {
                    // Before parsing the data, check if ROS is shutting down
                    if (ros::isShuttingDown()) {
                        socket_.close();
                        return; // Exit the handler early
                    }
                    parseData(std::string(recv_buffer_.begin(), recv_buffer_.begin() + bytes_transferred));
                    startReceive(); // Set up next receive operation
                } else {
                    ROS_ERROR("Receive error: %s", error.message().c_str());
                    // Optionally close the socket or stop receiving based on the type of error
                }
            });
    }
    void closeSocket() {
        socket_.close(); // Close the UDP socket
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& msg) {

            if (msg->header.stamp != last_update_time_){
                last_update_time_ = msg->header.stamp;
                sensor_msgs::LaserScan latest_scan = *msg; 
                // this->latest_scan.header.stamp = ros::Time::now(); 
                latest_scan.header.frame_id = "lidar_link"; 
                lidar_pub_.publish(latest_scan);
            }
            

    }

    private:
        ros::NodeHandle nh_;
        ros::Publisher joint_pub_, gt_pub_, lidar_pub_;
        ros::Subscriber lidar_sub_;
        udp::socket socket_;
        udp::endpoint remote_endpoint_;
        std::array<char, 1024> recv_buffer_;
        ros::Time last_update_time_;
        double gt_x = 0.0;
        double gt_y = 0.0;
        double gt_theta = 0.0;
        int counter=0;
        

        

        void parseData(const std::string& data) {
            if (data.size() < sizeof(unsigned long long) + 12 * sizeof(double)) {
                ROS_ERROR("Received data does not contain enough information.");
                return;
            }

            size_t offset = 0;

            // Unpack the unsigned long long (timestamp)
            unsigned long long timestamp;
            memcpy(&timestamp, data.data() + offset, sizeof(timestamp));
            offset += sizeof(timestamp);

            // Unpack the doubles
            double tokens[12];
            for (int i = 0; i < 12; ++i) {
                memcpy(&tokens[i], data.data() + offset, sizeof(double));
                offset += sizeof(double);
            }

            
            ROS_INFO_STREAM("Timestamp: " << timestamp);
            for (int i = 0; i < 12; ++i) {
                ROS_INFO_STREAM("Value[" << i << "]: " << tokens[i]);
            }

            if (counter==0){
                gt_x = tokens[6]/1000;
                gt_y=tokens[7]/1000;
                gt_theta=tokens[8]* (M_PI / 180.0);
                counter = 1;
            }

            double steerPose=tokens[2];
            double spinPose=tokens[3];
            double gtxPose=tokens[6]/1000;
            double gtyPose=tokens[7]/1000;
            double gtthetaPose=tokens[8]*(M_PI / 180.0);


            // Parsing joint state information
            sensor_msgs::JointState joint_state_msg;
            joint_state_msg.header.stamp = ros::Time::now();
            joint_state_msg.name.push_back("front_right_wheel_joint");
            joint_state_msg.position.push_back(std::isnan(spinPose) ? 0.0 : spinPose);
            
            joint_state_msg.name.push_back("front_right_steering_joint");
            joint_state_msg.position.push_back(std::isnan(steerPose) ? 0.0 : steerPose);


            


            double cosTheta = cos(-gt_theta);
            double sinTheta = sin(-gt_theta);
            double transformed_x = cosTheta * (gtxPose-gt_x) - sinTheta * (gtyPose-gt_y);
            double transformed_y = sinTheta * (gtxPose-gt_x) + cosTheta * (gtyPose-gt_y);
            double transformed_theta = gtthetaPose-gt_theta;

            

            transformed_theta = fmod(transformed_theta, 2*M_PI);
            if(transformed_theta>M_PI) transformed_theta-=2*M_PI;
            else if (transformed_theta<-M_PI) transformed_theta+=2*M_PI;
            
            // Parsing odometry information
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = ros::Time::now();
            odom_msg.header.frame_id = "world";
            odom_msg.child_frame_id = "robot_frame";
            odom_msg.pose.pose.position.x = transformed_x; // ground truth X
            odom_msg.pose.pose.position.y = transformed_y; // ground truth Y
            
            // Assuming theta represents a yaw angle, converting to quaternion for odometry message
            tf2::Quaternion q;
            q.setRPY(0, 0, transformed_theta); // ground truth THETA
            odom_msg.pose.pose.orientation = tf2::toMsg(q);

            // Publishing the messages
            joint_pub_.publish(joint_state_msg);
            gt_pub_.publish(odom_msg);
            
        
        }

    };

int main(int argc, char** argv) {
    ros::init(argc, argv, "data_parser2");

    try {
        boost::asio::io_service io_service;
        DataParser parser(io_service, "172.16.1.253", "5009");

        // Set up signal handling to stop the io_service on Ctrl+C
        boost::asio::signal_set signals(io_service, SIGINT);

        // Use a thread for io_service
        std::thread io_thread([&io_service](){
            io_service.run();
        });

        // Setup signal handler to properly stop io_service and the ROS node
        signals.async_wait([&io_service, &parser](const boost::system::error_code&, int) {
            parser.closeSocket();
            io_service.stop();
            ros::shutdown(); // Ensure ROS shutdown is called
        });

        ros::spin(); // This will now be executed immediately, allowing for proper ROS callback processing

        io_thread.join(); // Wait for the io_service thread to finish
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
