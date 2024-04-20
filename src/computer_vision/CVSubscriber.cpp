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

namespace CVParams {

inline bool running = false;

inline std::string WINDOW_NAME = "Practica Grupo 5";
inline std::string MODE = "Option [0-4]";

float PI = 3.14159265;
}

// -------- Self-Made Functions ----------------------------------------------
namespace CVFunctions {

// -------- Window Management Functions ----------------------------------------------
void initWindow()
{
  if (CVParams::running) return;
  CVParams::running = true;

  // Show images in a different windows
  cv::namedWindow(CVParams::WINDOW_NAME);
  // create Trackbar and add to a window
  cv::createTrackbar(CVParams::MODE, CVParams::WINDOW_NAME, nullptr, 4, 0);
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

  // Obtaining Parameter
  int mode_param = cv::getTrackbarPos(CVParams::MODE, CVParams::WINDOW_NAME);

  // Camera Intrinsic parameters
  float cx, cy, fx, fy;
  cx = camera_info_->k[2];
  cy = camera_info_->k[5];
  fx = camera_info_->k[0]; 
  fy = camera_info_->k[4]; 

  switch (mode_param)
  {
  case 1:
  {
    // As the frame of the pcl is the same as the camera, no rotation or translation
    cv::Mat rotation = (cv::Mat_<double>(3,3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    cv::Mat translation = (cv::Mat_<double>(3,1) << 0, 0, 0);

    // Building matrix
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    cv::Mat distCoeffs = (cv::Mat_<double>(4,1) << 0, 0, 0, 0);

    cv::Mat new_image(in_image_rgb.rows, in_image_rgb.cols, in_image_rgb.type());
    for (int i = 0; i < new_image.rows; i++) {
      for (int j = 0; j < new_image.cols; j++) {
        // You can now access the pixel value and calculate the new value
        new_image.at<cv::Vec3b>(i, j)[0] = 178;
        new_image.at<cv::Vec3b>(i, j)[1] = 178;
        new_image.at<cv::Vec3b>(i, j)[2] = 178;
      }
    }

    int X, Y;

    for (auto & point: in_pointcloud) {
      if (std::isinf(point.x) || std::isinf(point.y) || std::isinf(point.z)) {
        continue;
      }
      std::vector<cv::Point3f> target_point;
      std::vector<cv::Point2f> image_point;
      // Proyecting Point
      target_point.push_back(cv::Point3f(point.x, point.y, point.z));
      cv::projectPoints(target_point, rotation, translation, cameraMatrix, distCoeffs, image_point);

      X = (int)image_point[0].x;
      Y = (int)image_point[0].y;

      if (X >= new_image.cols || X < 0 || Y >= new_image.rows || Y < 0) {
        continue;
      }

      new_image.at<cv::Vec3b>(Y, X)[0] = point.b;
      new_image.at<cv::Vec3b>(Y, X)[1] = point.g;
      new_image.at<cv::Vec3b>(Y, X)[2] = point.r;
    }

    cv::medianBlur(new_image, new_image, 5);

    cv::imshow(CVParams::WINDOW_NAME, new_image);
    break;
  }
  case 2:
  {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_pcl (new pcl::PointCloud<pcl::PointXYZRGB>);
    // Fill in the cloud data
    new_pcl->width = in_image_depth.cols;
    new_pcl->height = in_image_depth.rows;
    new_pcl->is_dense = true;
    new_pcl->points.resize(new_pcl->width * new_pcl->height);

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
