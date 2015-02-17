#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <pcl_ros/point_cloud.h>

#include "cfg/cpp/kinect_differential/DiffKinectConfig.h"
#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/PointCloud2.h>
#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include "image_transport/image_transport.h"

const float fx_d = 5.9421434211923247e+02;
const float fy_d = 5.9104053696870778e+02;
const float cx_d = 3.3930780975300314e+02;
const float cy_d = 2.4273913761751615e+02;

class DepthImageHandler
{
public:
    DepthImageHandler(const std::string& name = "Kinect");
    ~DepthImageHandler();

    void handleDepthReceived(cv::Mat depth, cv::Mat XYZ);

    void messageCallback(kinect_differential::DiffKinectConfig& config, uint32_t level);
    void depthCallback(const sensor_msgs::ImageConstPtr &msg);
    void ptCloudCallback(const sensor_msgs::PointCloud2 &pc2);
    bool open(std::string depthTopic, std::string ptCloudTopic);



    cv::VideoCapture* cam;
    ros::NodeHandle _nh;
    ros::Publisher _pub;

    std::string _name;
    dynamic_reconfigure::Server<kinect_differential::DiffKinectConfig> _server;
    cv::Mat modelThresholdLow;
    cv::Mat modelThresholdHigh;
    float deviationThreshold;
    int numFramesToBuildModel;
    bool buildModel;
    image_transport::ImageTransport  it;
    image_transport::Subscriber imageSub;
    ros::Subscriber ptCloudSub;
    pcl::PointCloud<pcl::PointXYZ>::Ptr _cloudPtr;

};

DepthImageHandler::DepthImageHandler(const std::string &name):
    cam(NULL), it(_nh), _cloudPtr(new pcl::PointCloud<pcl::PointXYZ>)
{
    _name = name;
    std::cout << "Named resolved to: " << _name << std::endl;
    dynamic_reconfigure::Server<kinect_differential::DiffKinectConfig>::CallbackType f;
    f = boost::bind(&DepthImageHandler::messageCallback, this, _1, _2);
    _server.setCallback(f);
    _pub = _nh.advertise<sensor_msgs::PointCloud2>(_nh.resolveName("KinectCamera/PointCloud"),1); //_it.advertise(_name, 1);

    deviationThreshold = 4;
    numFramesToBuildModel = 50;
    buildModel = false;
}

DepthImageHandler::~DepthImageHandler()
{
    if(cam)
    {
        if(cam->isOpened())
            cam->release();
        delete cam;
    }
}

void DepthImageHandler::handleDepthReceived(cv::Mat depth, cv::Mat XYZ)
{
    if(buildModel)
    {
        static std::vector<cv::Mat> frames;
        frames.reserve(numFramesToBuildModel);
        if(frames.size() == numFramesToBuildModel)
        {
            cv::Mat sum = cv::Mat::zeros(depth.size(), CV_32F);
            cv::Mat sum_sq = cv::Mat::zeros(depth.size(), CV_32F);
            cv::Mat frame;
            for(int i = 0 ; i < frames.size(); ++i)
            {
                frames[i].convertTo(frame,CV_32F);
                sum += frame;
                sum_sq += frame.mul(frame);
            }
            cv::Mat mean = sum * (1.0 / frames.size());
            cv::Mat var = (sum_sq - sum.mul(sum)/float(frames.size()))/(float(frames.size()) - 1);
            cv::sqrt(var, var);
            cv::Mat(mean + 100).convertTo(modelThresholdHigh, CV_16U);
            cv::Mat(mean - 100).convertTo(modelThresholdLow, CV_16U);

            frames.clear();
            buildModel = false;
        }else
        {
            frames.push_back(depth);
        }

    }
    if(!modelThresholdHigh.empty() && !modelThresholdLow.empty())
    {
        if(_pub.getNumSubscribers() > 0)
        {
            cv::Mat mask1 = depth > modelThresholdHigh;
            cv::Mat mask2 = depth < modelThresholdLow;
            cv::Mat mask;

            cv::bitwise_or(mask1,mask2, mask);
            int numPoints = cv::countNonZero(mask);
            std::cout << "Number of filtered points: " << numPoints << std::endl;
            unsigned short* ptr = depth.ptr<unsigned short>(0);
            uchar* maskPtr = mask.ptr<uchar>(0);
            pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZ>());
            ptCloud->reserve(numPoints);
            unsigned int count = 0;
            for(int i = 0; i < mask.rows; ++i, ++ptr, ++maskPtr, ++count)
            {
                for(int j = 0; j < mask.cols; ++j, ++ptr, ++maskPtr, ++count)
                {
                    if(*maskPtr)
                    {
                        float depth = *ptr;
                        ptCloud->push_back((*_cloudPtr)[count]);
                    }
                }
            }
            _pub.publish(ptCloud);
        }
    }else
    {
        if(_pub.getNumSubscribers() == 0)
            return;
        std::cout << "Publishing point cloud with " << XYZ.rows << " " << XYZ.cols << std::endl;
        //pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZ>());
        //ptCloud->reserve(XYZ.rows*XYZ.cols);
        //unsigned short* ptr = depth.ptr<unsigned short>(0);
        //unsigned int count = 0;
        /*for(int i = 0; i < depth.rows; ++i, ++ptr, ++count)
        {
            for(int j = 0; j < depth.cols; ++j, ++ptr, ++count)
            {
                float depth = *ptr;
                ptCloud->push_back((*_cloudPtr)[count]);
            }
        }
        ptCloud->height = XYZ.rows;
        ptCloud->width = XYZ.cols;
        ptCloud->is_dense = true;*/
        _pub.publish(_cloudPtr);
    }
}

void DepthImageHandler::messageCallback(kinect_differential::DiffKinectConfig& config, uint32_t level)
{
    if(config.BuildModel == true)
    {
        std::cout << "Building model" << std::endl;
        buildModel = true;
        config.BuildModel = false;
    }
}

void DepthImageHandler::depthCallback(const sensor_msgs::ImageConstPtr &msg)
{
    cv::Mat depthImg;
    try
    {
        depthImg = cv_bridge::toCvCopy(msg, "16UC1")->image;
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to '16UC1'.", msg->encoding.c_str());
        return;
    }
    std::cout << "Received depth image of " << depthImg.rows << " x " << depthImg.cols << std::endl;
    handleDepthReceived(depthImg, cv::Mat());
}
void DepthImageHandler::ptCloudCallback(const sensor_msgs::PointCloud2 &pc2)
{

    pcl::fromROSMsg(pc2, *_cloudPtr);
    std::cout << "Received depth image: " << _cloudPtr->width << " x " << _cloudPtr->height << std::endl;
}

bool DepthImageHandler::open(std::string depthTopic, std::string ptCloudTopic)
{
    std::cout << "Subscribing to depth topic: " << depthTopic << std::endl;
    std::cout << "Subscribing to point topic: " << ptCloudTopic << std::endl;
    imageSub = it.subscribe(depthTopic, 1, &DepthImageHandler::depthCallback,this);
    ptCloudSub = _nh.subscribe(ptCloudTopic, 1, &DepthImageHandler::ptCloudCallback, this);

    ros::spin();
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "KinectCamera");

    DepthImageHandler dih(ros::names::remap("KinectCamera"));
    if(ros::names::remap("DepthTopic") == "DepthTopic")
    {
        ROS_WARN("DepthTopic has not been remapped!");
        return -1;
    }
    if(ros::names::remap("PointTopic") == "PointTopic")
        if(ros::names::remap("DepthTopic") == "DepthTopic")
        {
            ROS_WARN("PointTopic has not been remapped!");
            return -1;
        }
    dih.open(ros::names::remap("DepthTopic"), ros::names::remap("PointTopic"));
}
