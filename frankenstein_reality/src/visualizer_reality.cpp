#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>


class PathVisualizer {
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_, gt_sub_, slam_sub_;
    ros::Publisher odom_path_pub_, gt_path_pub_, slam_path_pub_;
    nav_msgs::Path odom_path_, gt_path_, slam_path_;

public:
    PathVisualizer() {
        // Initialize subscribers
        odom_sub_ = nh_.subscribe("/odom", 10, &PathVisualizer::odomCallback, this);
        gt_sub_ = nh_.subscribe("/ground_truth", 10, &PathVisualizer::gtCallback, this);
        slam_sub_ = nh_.subscribe("/tracked_pose", 10, &PathVisualizer::slamCallback, this);

        // Initialize publishers for visualizing paths
        odom_path_pub_ = nh_.advertise<nav_msgs::Path>("odom_path", 10, true);
        gt_path_pub_ = nh_.advertise<nav_msgs::Path>("gt_path", 10, true);
        slam_path_pub_ = nh_.advertise<nav_msgs::Path>("slam_path", 10, true);

        // Initialize paths
        odom_path_.header.frame_id = "world";
        gt_path_.header.frame_id = "world";
        slam_path_.header.frame_id = "world";
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        addPoseToPath(odom_path_, msg->pose.pose);
        odom_path_pub_.publish(odom_path_);
    }

    void gtCallback(const nav_msgs::Odometry::ConstPtr& msg) {
        addPoseToPath(gt_path_, msg->pose.pose);
        gt_path_pub_.publish(gt_path_);
    }

    void slamCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        addPoseStampedToPath(slam_path_, *msg);
        slam_path_pub_.publish(slam_path_);
    }

    void addPoseToPath(nav_msgs::Path& path, const geometry_msgs::Pose& pose) {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time::now();
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = pose;
        path.poses.push_back(pose_stamped);
    }

    void addPoseStampedToPath(nav_msgs::Path& path, const geometry_msgs::PoseStamped& pose_stamped) {
        path.poses.push_back(pose_stamped);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "path_visualizer");
    PathVisualizer visualizer;
    ros::spin();
    return 0;
}
