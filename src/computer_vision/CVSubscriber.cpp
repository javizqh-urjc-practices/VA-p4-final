/*
 Copyright (c) 2023 José Miguel Guerrero Hernández

 Licensed under the Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License;
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     https://creativecommons.org/licenses/by-sa/4.0/

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/*
Grupo 5: Javier Izquierdo y Sebastián Mayorquín
Partes implementadas:
- Opciones 1, 2 y 4
*/

#include "computer_vision/CVSubscriber.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>

namespace CVParams {

inline bool running = false;

inline std::string WINDOW_NAME = "Practica Grupo 5";
inline std::string MODE = "Option [0-4]";
inline std::string K = "K [1-5]";

cv::Ptr<cv::ml::KNearest> knn;

cv::Scalar lower_hsv(100, 30, 210);
cv::Scalar upper_hsv(130, 250, 255);

float PI = 3.14159265;
}

// -------- Self-Made Functions ----------------------------------------------
namespace CVFunctions {


cv::Mat preprocess(cv::Mat &image) {

    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat filtered_image, hsv_image, norm_image;

    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);
    cv::inRange(hsv_image, CVParams::lower_hsv, CVParams::upper_hsv, filtered_image);
    cv::morphologyEx(filtered_image, filtered_image, cv::MORPH_OPEN, element);
    cv::normalize(filtered_image, norm_image, 0, 1, cv::NORM_MINMAX);

    return filtered_image;
}

std::vector<cv::Point2f> generateDataFrame(cv::Mat &image){
    std::vector<cv::Point2f> points;
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<uchar>(y, x) == 255) {
                points.push_back(cv::Point2f(x, y));
            }
        }
    }
    return points;
}

std::vector<float> get_radius(cv::Mat& data, cv::Mat &labels, int K, cv::Mat centers){

    std::vector<float> radius(K, 0);
    for (int i = 0; i < data.rows; i++) {
        cv::Point2f pt = data.at<cv::Point2f>(i, 0);
        int idx = labels.at<int>(i, 0);
        float dist = cv::norm(pt - centers.at<cv::Point2f>(idx, 0));
        if (dist > radius[idx]) {
            radius[idx] = dist;
        }
    }
    return radius;
}
// -------- Window Management Functions ----------------------------------------------
void initWindow()
{
  if (CVParams::running) return;
  CVParams::running = true;

  // Show images in a different windows
  cv::namedWindow(CVParams::WINDOW_NAME);
  // create Trackbar and add to a window
  cv::createTrackbar(CVParams::MODE, CVParams::WINDOW_NAME, nullptr, 4, 0);
  cv::createTrackbar(CVParams::K, CVParams::WINDOW_NAME, nullptr, 4, 0);
}

}

namespace computer_vision
{

/**
   TO-DO: Default - the output images and pointcloud are the same as the input
 */
CVGroup CVSubscriber::processing(
  const cv::Mat in_image_rgb,
  const cv::Mat in_image_depth,
  const pcl::PointCloud<pcl::PointXYZRGB> in_pointcloud)
const
{
  // Create output images
  cv::Mat out_image_rgb, out_image_depth;
  // Create output pointcloud
  pcl::PointCloud<pcl::PointXYZRGB> out_pointcloud;

  // Processing
  out_image_rgb = in_image_rgb;
  out_image_depth = in_image_depth;
  out_pointcloud = in_pointcloud;

  // First time execution
  CVFunctions::initWindow();

  cv::Mat image, filtered_image;
  std::vector<cv::Point2f> points;
  cv::Mat labels, data, centers;
  std::vector<float> radius;
  // Obtaining Parameter

  int mode_param = cv::getTrackbarPos(CVParams::MODE, CVParams::WINDOW_NAME);
  int K = cv::getTrackbarPos(CVParams::K, CVParams::WINDOW_NAME) + 1;

  switch (mode_param)
  {
  case 1:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  case 2:
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_pcl (new pcl::PointCloud<pcl::PointXYZRGB>);
    // Fill in the cloud data
    new_pcl->width = in_image_depth.cols;
    new_pcl->height = in_image_depth.rows;
    new_pcl->is_dense = true;
    new_pcl->points.resize(new_pcl->width * new_pcl->height);

    float cx, cy, fx, fy;//principal point and focal lengths
    cx = camera_info_->k[2];
    cy = camera_info_->k[5];
    fx = camera_info_->k[0]; 
    fy = camera_info_->k[4]; 

    int depth_idx = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::iterator pt_iter = new_pcl->begin ();
    for (int v = 0; v < (int)new_pcl->height; ++v) {
      for (int u = 0; u < (int)new_pcl->width; ++u, ++depth_idx, ++pt_iter) {
        pcl::PointXYZRGB& pt = *pt_iter;
        float Z = in_image_depth.at<float>(v, u);
        // Check for invalid measurements
        if (std::isnan (Z)) {
          pt.x = pt.y = pt.z = Z;
        } else {
          pt.x = ((u - cx) * Z) / fx;
          pt.y = ((v - cy) * Z) / fy;
          pt.z = Z;
          pt.r = in_image_rgb.at<cv::Vec3b>(v,u)[2];
          pt.g = in_image_rgb.at<cv::Vec3b>(v,u)[1];
          pt.b = in_image_rgb.at<cv::Vec3b>(v,u)[0];
          // std::cout << rgb_image.at<cv::Vec3b>(i,j)[0] << std::endl;
        }
      }
    }

    out_pointcloud = *new_pcl;
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  }
  case 3:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  
  case 4:

    filtered_image = CVFunctions::preprocess(out_image_rgb);
    points = CVFunctions::generateDataFrame(filtered_image);

    data = cv::Mat(points.size(), 1, CV_32FC2, &points[0]);

    // Using K-means
    cv::kmeans(data, K, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, centers);
    
    radius = CVFunctions::get_radius(data, labels, K, centers);

    // Drawing
    for (int i = 0; i < K; i++) {
        cv::circle(out_image_rgb, cv::Point(centers.at<cv::Point2f>(i, 0).x, centers.at<cv::Point2f>(i, 0).y),
                   radius[i], cv::Scalar(0, 0, 255), 2);
        cv::circle(out_image_rgb, cv::Point(centers.at<cv::Point2f>(i, 0).x, centers.at<cv::Point2f>(i, 0).y),
                   3, cv::Scalar(0, 255, 0), -1); // Centroide en verde
    }
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;  

  default:
    // Show unprocessed image and point cloud
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  }

  cv::waitKey(3);

  return CVGroup(out_image_rgb, out_image_depth, out_pointcloud);
}

} // namespace computer_vision
