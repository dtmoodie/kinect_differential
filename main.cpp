#include <ros/ros.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>
#include <map>
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

#include "gloox/disco.h"
#include "gloox/message.h"
#include "gloox/gloox.h"
#include "gloox/siprofileft.h"
#include "gloox/siprofilefthandler.h"
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
using namespace gloox;


const float fx_d = 5.9421434211923247e+02;
const float fy_d = 5.9104053696870778e+02;
const float cx_d = 3.3930780975300314e+02;
const float cy_d = 2.4273913761751615e+02;
#define NUM_THREADS 4

struct BoundingBox;
struct Triggerhandler
{
  virtual void handleTrigger() = 0;
};
struct Parameter
{
  Parameter(BoundingBox* parent_ = NULL): parent(parent_){}
  virtual void clear() = 0;
  virtual void accumulate(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, int idx) = 0;
  virtual double getValue() = 0;
  BoundingBox* parent;
};

struct BoundingBox
{
  double minX, maxX;
  double minY, maxY;
  double minZ, maxZ;

  bool process(const pcl::PointCloud<pcl::PointXYZ>& pointCloud)
  {
    for(int i = 0; i < first_pass_parameters.size(); ++i)
    {
      first_pass_parameters[i]->clear();
    }
    std::vector<int> idx;
    for(int i = 0; i < pointCloud.points.size(); ++i)
    {
      if(pointCloud.points[i].x > minX && pointCloud.points[i].x < maxX &&
         pointCloud.points[i].y > minY && pointCloud.points[i].y < maxY &&
         pointCloud.points[i].z > maxZ && pointCloud.points[i].z < maxZ)
      {
        for(int j = 0; j < first_pass_parameters.size(); ++j)
        {
          first_pass_parameters[j]->accumulate(pointCloud, i);
        }
        idx.push_back(i);
      }
    }
    for(int i = 0; i < first_pass_parameters.size(); ++i)
    {
      first_pass_parameters[i]->getValue();
    }
    for(int i = 0; i < second_pass_parameters.size(); ++i)
    {
      second_pass_parameters[i]->clear();
    }
    for(int i = 0; i < idx.size(); ++i)
    {
      for(int j = 0; j < second_pass_parameters.size(); ++j)
      {
        second_pass_parameters[j]->accumulate(pointCloud, idx[i]);
      }
    }
    for(int i = 0; i < second_pass_parameters.size(); ++i)
    {
      second_pass_parameters[i]->getValue();
    }
    bool pass = true;
    for(std::map<std::string, double>::iterator itr = values.begin(); itr != values.end();  ++itr)
    {
      if(itr->second > thresholds[itr->first])
        pass = false;
    }
    return pass;
  }

  std::map<std::string, double> values;
  std::vector<Parameter*> first_pass_parameters;
  std::vector<Parameter*> second_pass_parameters;
  std::map<std::string, double> thresholds;
};
struct Centroid: public Parameter
{
  Centroid(int dim_, BoundingBox* parent_ = NULL):Parameter(parent_), dim(dim_){}
  void clear()
  {
    sum = 0; count = 0;
  }
  void accumulate(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, int idx)
  {
    switch (dim)
    {
      case(0): sum += pointCloud.points[idx].x; break;
      case(1): sum += pointCloud.points[idx].y; break;
      case(2): sum += pointCloud.points[idx].z; break;
    }
    ++count;
  }
  double getValue()
  {
    sum /= count;
    parent->values["C" + boost::lexical_cast<std::string>(dim)] = sum;
    return sum;
  }
  double sum;
  int count;
  int dim;
};

struct Moment: public Parameter
{
  Moment(int Px_, int Py_, int Pz_, BoundingBox* parent_):Parameter(parent_), Px(Px_), Py(Py_), Pz(Pz_){}
  void clear()
  {
    Cx = parent->values["C0"];
    Cy = parent->values["C1"];
    Cz = parent->values["C2"];
    value = 0;
  }
  void accumulate(const pcl::PointCloud<pcl::PointXYZ>& pointCloud, int idx)
  {
    value += pow(pointCloud.points[idx].x - Cx, Px) + pow(pointCloud.points[idx].y - Cy, Py) + pow(pointCloud.points[idx].z - Cz, Pz);
  }
  double getValue()
  {
    parent->values["HU" + boost::lexical_cast<std::string>(Px) + boost::lexical_cast<std::string>(Py) + boost::lexical_cast<std::string>(Cz)] = value;
    return value;
  }
  double Cx, Cy, Cz;
  int Px, Py, Pz;
  double value;
};

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

    // Gloox handlers
    virtual void onConnect();
    virtual void onDisconnect(ConnectionError e);
    virtual bool onTLSConnect(const CertInfo& info);
    virtual void handleMessage(const Message& msg, MessageSession * session);
    virtual void handleMessageEvent(const JID& from, MessageEventType messageEvent);
    virtual void handleChatState(const JID& from, ChatStateType state);
    virtual void handleMessageSession(MessageSession *session);
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
    pcl::PointCloud<pcl::PointXYZ>::Ptr _filteredCloudPtr;
    boost::mutex mtx;
    pcl::search::FlannSearch<pcl::PointXYZ>::Ptr _modelTree;
    std::vector<BoundingBox> boundingBoxes;
    // Gloox
    std::shared_ptr<gloox::Client> xmpp_client;
    MessageSession *m_session;
    MessageEventFilter* m_messageEventFilter;
    ChatStateFilter* m_chatStateFilter;
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
    {
       if(_cloudPtr->points[i].x == _cloudPtr->points[i].x)
       {
          idx.push_back(i);
       }
    }
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
