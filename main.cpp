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
#include <boost/thread/mutex.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <cv_bridge/cv_bridge.h>
#include "image_transport/image_transport.h"
#include "gloox/disco.h"
#include "gloox/message.h"
#include "gloox/gloox.h"
#include "gloox/siprofileft.h"
#include "gloox/bytestreamdatahandler.h"
#include "gloox/socks5bytestreamserver.h"
#include "gloox/loghandler.h"
#include "gloox/connectionlistener.h"
#include "gloox/messagesessionhandler.h"
#include "gloox/messageeventhandler.h"
#include "gloox/messageeventfilter.h"
#include "gloox/messagehandler.h"
#include "gloox/client.h"
#include "gloox/chatstatehandler.h"
#include "gloox/chatstatefilter.h"

const float fx_d = 5.9421434211923247e+02;
const float fy_d = 5.9104053696870778e+02;
const float cx_d = 3.3930780975300314e+02;
const float cy_d = 2.4273913761751615e+02;

using namespace gloox;
class DepthImageHandler: public MessageSessionHandler, ConnectionListener, LogHandler, MessageEventHandler, MessageHandler, ChatStateHandler
{
public:
    DepthImageHandler(const std::string& name = "Kinect");
    ~DepthImageHandler();

    void handleDepthReceived(cv::Mat depth, cv::Mat XYZ);

    void messageCallback(kinect_differential::DiffKinectConfig& config, uint32_t level);
    void depthCallback(const sensor_msgs::ImageConstPtr &msg);
    void ptCloudCallback(const sensor_msgs::PointCloud2 &pc2);
    bool open(std::string depthTopic, std::string ptCloudTopic);
    
    // gloox message handling functions
    virtual void onCOnnect();
    virtual void onDisconnect(ConnectionError e);
    virtual bool onTLSConnect(const CertInfo& info)
    virtual void handleMessage(const Message& msg, MessageSession* session);
    virtual void handleMessageEvent(const JID& from, MessageEventType messageEvent);
    virtual void handleChatState(const JID& from, ChatStateType state);
    virtual void handleMessageSession(MessageSession* session);
    virtual void handleLog(LogLevel level, LogArea area, const std::string& message);
    
    cv::VideoCapture* cam;
    ros::NodeHandle _nh;
    ros::Publisher _pub;
    ros::Publisher _modelPub;

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
    pcl::PointCloud<pcl::PointXYZ>::Ptr _modelPtr;
    boost::mutex mtx;
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
    numFramesToBuildModel = 10;
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
boost::mutex::scoped_lock lock(mtx);
std::cout << "Received depth image " << depth.depth()<< std::endl;
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
                //sum_sq += frame.mul(frame);
            }
            cv::Mat mean = sum * (1.0 / frames.size());
            //cv::Mat var = (sum_sq - sum.mul(sum)/float(frames.size()))/(float(frames.size()) - 1);
            //cv::sqrt(var, var);
            cv::Mat(mean + 10).convertTo(modelThresholdHigh, CV_16U);
            cv::Mat(mean - 10).convertTo(modelThresholdLow, CV_16U);

            frames.clear();
            buildModel = false;
        }else
        {
            frames.push_back(depth);
        }
        _modelPtr = _cloudPtr;
        _cloudPtr.reset(new pcl::PointCloud<pcl::PointXYZ>);
        _modelPub = _nh.advertise<sensor_msgs::PointCloud2>(_nh.resolveName("KinectCamera/ModelCloud"),1); //_it.advertise(_name, 1);
        _modelPub.publish(_modelPtr);
      std::cout << "Publishing model " << std::endl;
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
            if(numPoints)
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
                        ptCloud->push_back((*_cloudPtr)[count]);
                    }
                }
            }
            _pub.publish(ptCloud);
         std::cout << "Publishing filtered cloud" << std::endl;
		return;
        }
    }else
    {
        if(_pub.getNumSubscribers() == 0)
            return;
//static boost::posix_time::ptime lastTIme = boost::date_time::microsec_clock<boost::posix_time::ptime>::universal_time();
//	boost::posix_time::ptime currentTime = boost::date_time::microsec_clock<boost::posix_time::ptime>::unviersal_time();
//if(boost::posix_time::time_duration(currentTime - lastTime).total_milliseconds() > 300)
//{/
//	_pub.publish(_cloudPtr);
//	lastTime = boost::date_time::microsec_clock<boost::posix_time::ptime>::unviersal_time();
//}
//else
//{
	
//}
//        std::cout << "Publishing point cloud with " << XYZ.rows << " " << XYZ.cols << std::endl;
//        _pub.publish(_cloudPtr);
//    std::cout << "Publishing unfiltered cloud " << std::endl;

static int count = 0;
if(count == 10)
{
_pub.publish(_cloudPtr);
count = 0;
}
++count;

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
    if(config.ClearModel == true)
    {
     std::cout << "Clearning model" << std::endl;
     config.ClearModel = false;
      boost::mutex::scoped_lock lock(mtx);
      modelThresholdHigh.release();
     modelThresholdLow.release();
    std::cout << "Finished clearing model" << std::endl;
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
  //  std::cout << "Received depth image of " << depthImg.rows << " x " << depthImg.cols << std::endl;
    handleDepthReceived(depthImg, cv::Mat());
}
void DepthImageHandler::ptCloudCallback(const sensor_msgs::PointCloud2 &pc2)
{

    pcl::fromROSMsg(pc2, *_cloudPtr);
    //std::cout << "Received depth image: " << _cloudPtr->width << " x " << _cloudPtr->height << std::endl;
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
    {
        ROS_WARN("PointTopic has not been remapped!");
        return -1;
    }
    dih.open(ros::names::remap("DepthTopic"), ros::names::remap("PointTopic"));
}
