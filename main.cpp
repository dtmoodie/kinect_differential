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

#include <pcl/search/flann_search.h>
#include <pcl/search/impl/flann_search.hpp>
#include <pcl/search/kdtree.h>

const float fx_d = 5.9421434211923247e+02;
const float fy_d = 5.9104053696870778e+02;
const float cx_d = 3.3930780975300314e+02;
const float cy_d = 2.4273913761751615e+02;
#define NUM_THREADS 4
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
	pcl::PointCloud<pcl::PointXYZ>::Ptr _filteredCloudPtr;
    boost::mutex mtx;
	
	pcl::search::FlannSearch<pcl::PointXYZ>::Ptr _modelTree;
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
void radiusSearchHelper(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::search::Search<pcl::PointXYZ>::Ptr searchTree, const std::vector<int>& idx, std::vector<unsigned char>* found, int threads, int thread, float radius)
{
    std::vector<int> knn_idx;
    std::vector<float> dist;
    for(int i = thread; i < idx.size(); i+= threads)
    {
        searchTree->radiusSearch(*cloud, idx[i], radius, knn_idx,dist,1);
        (*found)[i] = knn_idx.size();
    }
}
void DepthImageHandler::handleDepthReceived(cv::Mat depth, cv::Mat XYZ)
{
	boost::mutex::scoped_lock lock(mtx);
	std::cout << "Received depth image " << depth.depth()<< std::endl;
    if(_modelTree)
	{
		std::vector<int> idx;

        idx.reserve(_cloudPtr->points.size());
        for(int i = _cloudPtr->points.size() - 1; i > 0; --i)
            if(_cloudPtr->points[i].x == _cloudPtr->points[i].x)
                idx.push_back(i);
		std::vector<unsigned char> found(idx.size(),0);
        boost::thread threads[NUM_THREADS - 1];
        for(int i = 1; i < NUM_THREADS; ++i)
        {
            threads[i - 1] = boost::thread(boost::bind(&radiusSearchHelper,_cloudPtr, _modelTree, boost::ref(idx), &found, NUM_THREADS, i, 0.2));
        }
		radiusSearchHelper(_cloudPtr, _modelTree, idx, &found, NUM_THREADS, 0, 0.2);
        for(int i = 0; i < NUM_THREADS - 1; ++i)
        {
            threads[i].join();
        }
        size_t count = std::accumulate(found.begin(), found.end(), 0);
        _filteredCloudPtr.reset(new pcl::PointCloud<pcl::PointXYZ>());
        _filteredCloudPtr->points.reserve(_cloudPtr->points.size() - count);
        for(int i = 0; i < found.size(); ++i)
        {
            if(found[i] == 0)
                _filteredCloudPtr->points.push_back(_cloudPtr->points[idx[i]]);
        }
		_pub.publish(_filteredCloudPtr);
	}else
	{
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
		_modelPtr = _cloudPtr;
		_cloudPtr.reset(new pcl::PointCloud<pcl::PointXYZ>);
		_modelTree.reset(new pcl::search::FlannSearch<pcl::PointXYZ>(true));
        _modelTree->setInputCloud(_modelPtr);
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
