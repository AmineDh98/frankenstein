#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>

class ScanFilterNode
{
public:
    ScanFilterNode()
    {
        scan_sub = nh.subscribe<sensor_msgs::LaserScan>("/scan", 10, &ScanFilterNode::scanCallback, this);
        cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/filtered_cloud", 10);
    }

    void scanCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
    {
        
        sensor_msgs::PointCloud2 cloud;
        projector.projectLaser(*scan, cloud);

        
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(cloud, *pcl_cloud);

        // Apply Statistical Outlier Removal Filter
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        sor.setInputCloud(pcl_cloud);
        sor.setMeanK(10); // Number of neighbors to analyze for each point
        sor.setStddevMulThresh(0.5); // Standard deviation multiplier
        sor.filter(*filtered_cloud);

        
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*filtered_cloud, output);
        output.header.frame_id = scan->header.frame_id;
        cloud_pub.publish(output);
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber scan_sub;
    ros::Publisher cloud_pub;
    laser_geometry::LaserProjection projector;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "scan_filter_node");
    ScanFilterNode node;
    ros::spin();
    return 0;
}
