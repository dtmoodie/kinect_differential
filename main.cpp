#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <iostream>
#include <pcl_ros/point_cloud.h>

#include "cfg/cpp/kinect_differential/DiffKinectConfig.h"
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <boost/thread.hpp>
#include <boost/date_time.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

class DepthImageHandler
{
public:
    DepthImageHandler(const std::string& name = "Kinect");
    ~DepthImageHandler();

    void handleDepthReceived(cv::Mat depth, cv::Mat XYZ);

    void messageCallback(kinect_differential::DiffKinectConfig& config, uint32_t level);

    bool open();




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
};

DepthImageHandler::DepthImageHandler(const std::string &name):
    cam(NULL)
{
    _name = _nh.resolveName(name);
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
            cv::imshow("Mask1", mask1);
            cv::imshow("Mask2", mask2);
            cv::Mat mask;

            cv::bitwise_or(mask1,mask2, mask);
            cv::imshow("Mask", mask);
            int numPoints = cv::countNonZero(mask);
            std::cout << "Number of filtered points: " << numPoints << std::endl;
            cv::Vec3f* ptr = XYZ.ptr<cv::Vec3f>(0);
            uchar* maskPtr = mask.ptr<uchar>(0);
            pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZ>());
            ptCloud->reserve(numPoints);
            for(int i = 0; i < mask.rows*mask.cols; ++i, ++ptr, ++maskPtr)
            {
                if(*maskPtr)
                {
                    ptCloud->push_back(pcl::PointXYZ(ptr->val[0], ptr->val[1], ptr->val[2]));
                }
            }
            _pub.publish(ptCloud);
        }
    }else
    {
        if(_pub.getNumSubscribers() == 0)
            return;
        std::cout << "Publishing point cloud with " << XYZ.rows << " " << XYZ.cols << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr ptCloud(new pcl::PointCloud<pcl::PointXYZ>());
        ptCloud->reserve(XYZ.rows*XYZ.cols);
        cv::Vec3f* ptr = XYZ.ptr<cv::Vec3f>(0);
        for(int i = 0; i < XYZ.rows*XYZ.cols; ++i, ++ptr)
        {
            ptCloud->push_back(pcl::PointXYZ(ptr->val[0], ptr->val[1], ptr->val[2]));
        }
        ptCloud->height = XYZ.rows;
        ptCloud->width = XYZ.cols;
        ptCloud->is_dense = true;
        _pub.publish(ptCloud);
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

bool DepthImageHandler::open()
{
    if(cam == NULL)
        cam = new cv::VideoCapture();
    cam->open(cv::CAP_OPENNI);
    if(cam->isOpened())
    {
        std::cout << "Successfully opened camera" << std::endl;
    }
    int key = 0;
    cv::Mat depth, XYZ;
    while(_nh.ok() && key != 27)
    {
        if(cam->read(depth))
            cv::imshow("Depth", depth);
        if(cam->retrieve(XYZ, cv::CAP_OPENNI_POINT_CLOUD_MAP))
            cv::imshow("XYZ", XYZ);
        key = cv::waitKey(30);

        handleDepthReceived(depth,XYZ);
        //boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
        ros::spinOnce();
    }
}



int main(int argc, char** argv)
{
    ros::init(argc, argv, "KinectCamera");
    DepthImageHandler dih("KinectCamera");
    dih.open();
}
