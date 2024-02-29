#include <gazebo/gazebo.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/Vector3.hh>

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/JointState.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_broadcaster.h>


namespace gazebo {
class GazeboRos3WheelDrive : public ModelPlugin {
public:
    GazeboRos3WheelDrive() : ModelPlugin() {
        printf("Control Plugin Created!\n");
    }

    void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) override {
        this->model = _model;

        if (!ros::isInitialized()) {
            ROS_FATAL_STREAM("A ROS node for Gazebo has not been initialized, unable to load plugin.");
            return;
        }
        ROS_INFO("ROS Model Plugin Loaded!");
        std::cout << "Model Name = " << _model->GetName() << std::endl;
        this->last_update_time_ = this->model->GetWorld()->SimTime();
        // this->odom_pub = this->nh.advertise<nav_msgs::Odometry>("/odom", 10);
        this->joints_pub = this->nh.advertise<sensor_msgs::JointState>("/frankenstein/joint_states", 10);
        this->cmd_subscriber = this->nh.subscribe("/cmd_vel", 10, &GazeboRos3WheelDrive::call_back, this);
        this->ground_truth_subscriber = this->nh.subscribe("/ground_truth", 10, &GazeboRos3WheelDrive::call_back_ground_truth, this);
        
        
        // Connect to the update event of the simulation
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&GazeboRos3WheelDrive::OnUpdate, this, std::placeholders::_1));
        
        this->wheel_radius = 0.1;

        // this->Wheelbase = 1.5;
        // this->SteeringWheelOffset = 0.2;

        this->steering_gain = 0.1; 
        
        this->joint1 = this->model->GetJoint("front_right_wheel_joint");
        this->joint2 = this->model->GetJoint("front_right_steering_joint");

        if (!joint1 || !joint2) {
            ROS_FATAL_STREAM("Joints not initialized.");
            return;
        }
        this->counter = 0;  
        // publishTransform();        
    }

    void call_back(const geometry_msgs::Twist::ConstPtr& msg) {
        this->desired_linear_velocity = msg->linear.x;
        this->desired_angular_velocity = this->desired_linear_velocity/this->wheel_radius;
        this->desired_angle = msg->angular.z;
        
        
    }

    void call_back_ground_truth(const nav_msgs::Odometry::ConstPtr& msg) {
        this->gt_x = msg->pose.pose.position.x;
        this->gt_y = msg->pose.pose.position.y;
        tf2::Quaternion q(
        msg->pose.pose.orientation.x,
        msg->pose.pose.orientation.y,
        msg->pose.pose.orientation.z,
        msg->pose.pose.orientation.w
        );
        tf2::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);
        this->gt_orientation = yaw;
        
    }

    void OnUpdate(const common::UpdateInfo & /*info*/) {
        common::Time current_time = this->model->GetWorld()->SimTime(); // Use Gazebo's sim time
        common::Time duration = current_time - this->last_update_time_;
        // Convert the duration to a double representing seconds
        double dt = duration.Double();
        
        
        // if (dt>=(1/10)){
            
        this->last_update_time_ = current_time;         
        if((this->desired_linear_velocity!=0.0)&&(this->counter==0)){
            this->theta = this->gt_orientation;
            this->x = this->gt_x;
            this->y = this->gt_y;
            this->counter+=1;

        }
            
        joint1->SetVelocity(0, desired_angular_velocity); 


        double current_steering_angle = joint2->Position(0);
        double steering_error = this->desired_angle - current_steering_angle;
        double steering_velocity = this->steering_gain * steering_error;

        joint2->SetVelocity(0, steering_velocity);

        if (std::abs(steering_error) < 1e-10) { 
            joint2->SetVelocity(0, 0.0); 
        }
            
        publishJointStates();
            // double current_angular_velocity = joint1->GetVelocity(0);
            // double current_linear_velocity = current_angular_velocity*this->wheel_radius ;

        
            // double R_offset = (this->Wheelbase  / tan(current_steering_angle)) + this->SteeringWheelOffset;
            // // Calculate angular velocity of the vehicle
            // double omega = current_linear_velocity / R_offset; 
            // double deltaTheta = omega * dt;
            // this->theta += deltaTheta;
            // // Ensure theta is within -PI to PI range
            // this->theta = atan2(sin(this->theta), cos(this->theta));
            
            // double deltaX, deltaY;
            // if (fabs(current_steering_angle) == 0) { // Check for straight movement
            //     deltaX = current_linear_velocity * cos(this->theta) * dt;
            //     deltaY = current_linear_velocity * sin(this->theta) * dt;
            // } else { // For turning, adjust calculations to reflect arc movement
            //     deltaX = R_offset * sin(this->theta + deltaTheta) - R_offset * sin(this->theta);
            //     deltaY = -R_offset * cos(this->theta + deltaTheta) + R_offset * cos(this->theta);
            // }
            
            // this->x += deltaX ; 
            // this->y += deltaY ;

            // this->vx = deltaX/dt; 
            // this->vy = deltaY/dt; 
            // this->vth = omega; 


            
            // publishOdometry();
            // publishTransform();
            // this->last_update_time_ = current_time;
        
        
    }

private:
    physics::ModelPtr model; 
    ros::NodeHandle nh;
    ros::Subscriber cmd_subscriber;
    ros::Subscriber ground_truth_subscriber;
    ros::Publisher joints_pub;
    // ros::Publisher odom_pub;
    common::Time last_update_time_;
    event::ConnectionPtr updateConnection;
    double desired_linear_velocity = 0.0;
    double desired_angular_velocity = 0.0;

    double desired_angle = 0.0;
    double wheel_radius;

    // double Wheelbase;
    // double SteeringWheelOffset;

    double steering_gain; 
    
    // double vx = 0.0;
    // double vy = 0.0;
    // double vth = 0.0;


    physics::JointPtr joint1;
    physics::JointPtr joint2;

    double x = 0.0, y = 0.0, theta = 0.0, gt_x = 0.0, gt_y = 0.0, gt_orientation = 0.0;
    // tf2_ros::TransformBroadcaster tf_broadcaster;

    double counter;
    ros::Time previous;

    
    // void publishTransform() {
    //       // Broadcast the transformation
    //     geometry_msgs::TransformStamped transformStamped;
    //     transformStamped.header.stamp = ros::Time::now();
    //     transformStamped.header.frame_id = "odom";
    //     transformStamped.child_frame_id = "robot_frame";
    //     transformStamped.transform.translation.x = this->x; 
    //     transformStamped.transform.translation.y = this->y;
    //     transformStamped.transform.translation.z = 0.0;
    //     tf2::Quaternion q;
    //     q.setRPY(0, 0, this->theta); // Set rotation as quaternion
    //     transformStamped.transform.rotation.x = q.x();
    //     transformStamped.transform.rotation.y = q.y();
    //     transformStamped.transform.rotation.z = q.z();
    //     transformStamped.transform.rotation.w = q.w();

    //     this->tf_broadcaster.sendTransform(transformStamped);
    // }

    // void publishOdometry() {
    //     // Publish odometry
    //     nav_msgs::Odometry odom_msg;
    //     odom_msg.header.stamp = ros::Time::now();
    //     odom_msg.header.frame_id = "odom";
    //     odom_msg.child_frame_id = "robot_frame";
    //     odom_msg.pose.pose.position.x = this->x; 
    //     odom_msg.pose.pose.position.y = this->y; 
    //     odom_msg.pose.pose.orientation = tf::createQuaternionMsgFromYaw(this->theta); 
    //     odom_msg.twist.twist.linear.x = this->vx;
    //     odom_msg.twist.twist.linear.y = this->vy;
    //     odom_msg.twist.twist.angular.z = this->vth;
    //     odom_pub.publish(odom_msg);

    // }

    void publishJointStates() {
        sensor_msgs::JointState joint_state_msg;
        ros::Time now = ros::Time::now();
        if (this->previous != now){
            joint_state_msg.header.stamp = now;
            joint_state_msg.name.push_back("front_right_wheel_joint");
            joint_state_msg.name.push_back("front_right_steering_joint");
            joint_state_msg.position.push_back(this->joint1->Position(0));
            joint_state_msg.position.push_back(this->joint2->Position(0));
            joint_state_msg.velocity.push_back(this->joint1->GetVelocity(0));
            joint_state_msg.velocity.push_back(this->joint2->GetVelocity(0));
            joints_pub.publish(joint_state_msg);
            this->previous = now;
        }
        
    }
    
    
};

GZ_REGISTER_MODEL_PLUGIN(GazeboRos3WheelDrive)
}
