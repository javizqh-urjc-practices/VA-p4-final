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
  
  // cv::createTrackbar("h", CVParams::WINDOW_NAME, nullptr, 255, 0);
  // cv::createTrackbar("s", CVParams::WINDOW_NAME, nullptr, 255, 0);
  // cv::createTrackbar("v", CVParams::WINDOW_NAME, nullptr, 255, 0);
  // cv::createTrackbar("h2", CVParams::WINDOW_NAME, nullptr, 255, 0);
  // cv::createTrackbar("s2", CVParams::WINDOW_NAME, nullptr, 255, 0);
  // cv::createTrackbar("v2", CVParams::WINDOW_NAME, nullptr, 255, 0);
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

  switch (mode_param)
  {
  case 1:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  case 2:
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
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
