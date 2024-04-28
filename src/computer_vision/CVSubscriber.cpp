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

#include <sys/time.h>

#include <chrono>
#include <ctime>
#include <stdio.h>
#include <fmt/format.h>


namespace CVParams {

inline bool running = false;

inline std::string WINDOW_NAME = "Practica Grupo 5";
inline std::string MODE = "Option [0-4]";
inline std::string K = "K [1-5]";

inline bool is_depth_in_meters = false;

struct timeval last_time {};

cv::Ptr<cv::ml::KNearest> knn;

cv::Scalar lower_hsv(100, 30, 210);
cv::Scalar upper_hsv(130, 250, 255);

cv::Mat old_3_gray;
std::vector<cv::Point2f> points_3_old;

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
void initWindow(cv::Mat in_image_rgb)
{
  if (CVParams::running) return;
  CVParams::running = true;

  // Show images in a different windows
  cv::namedWindow(CVParams::WINDOW_NAME);
  // create Trackbar and add to a window
  cv::createTrackbar(CVParams::MODE, CVParams::WINDOW_NAME, nullptr, 4, 0);
  cv::createTrackbar(CVParams::K, CVParams::WINDOW_NAME, nullptr, 4, 0);
  gettimeofday(&CVParams::last_time, nullptr);
  // Take first frame and find corners in it
  cv::cvtColor(in_image_rgb, CVParams::old_3_gray, cv::COLOR_BGR2GRAY);
  cv::goodFeaturesToTrack(CVParams::old_3_gray, CVParams::points_3_old, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
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
  CVFunctions::initWindow(in_image_rgb);
  if (in_image_depth.type() == CV_32F) CVParams::is_depth_in_meters = true;

  cv::Mat image, filtered_image;
  std::vector<cv::Point2f> points;
  cv::Mat labels, data, centers;
  std::vector<float> radius;
  // Obtaining Parameter

  int mode_param = cv::getTrackbarPos(CVParams::MODE, CVParams::WINDOW_NAME);
  int K = cv::getTrackbarPos(CVParams::K, CVParams::WINDOW_NAME) + 1;

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
    cv::Mat bool_image(in_image_rgb.rows, in_image_rgb.cols, CV_8UC1);
    for (int i = 0; i < new_image.rows; i++) {
      for (int j = 0; j < new_image.cols; j++) {
        // You can now access the pixel value and calculate the new value
        new_image.at<cv::Vec3b>(i, j)[0] = 0;
        new_image.at<cv::Vec3b>(i, j)[1] = 0;
        new_image.at<cv::Vec3b>(i, j)[2] = 0;
        bool_image.at<uchar>(i, j) = 0;
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
      bool_image.at<uchar>(Y,X) = 1;
    }

    bool found_match;
    int k_size = 1;
    int n_match = 0;
    int match_r, match_g, match_b;

    for (int i = 0; i < new_image.rows; i++) {
      for (int j = 0; j < new_image.cols; j++) {
        if ((uint)bool_image.at<uchar>(i, j) == 0) {
          found_match = false;
          k_size = 1;
          n_match = 0;
          match_r = 0;
          match_g = 0;
          match_b = 0;
          while (!found_match) {
            for (int k = - k_size; k <= k_size; k++) {
              for (int l = - k_size; l <= k_size; l++) {
                if (l == 0 && k == 0) continue;
                if (i+l < 0 || j+k < 0) continue;
                if (i+l > new_image.rows || j+k > new_image.cols) continue;
                if ((uint)bool_image.at<uchar>(i+l,j+k) == 1) {
                  match_b += new_image.at<cv::Vec3b>(i+l, j+k)[0];
                  match_g += new_image.at<cv::Vec3b>(i+l, j+k)[1];
                  match_r += new_image.at<cv::Vec3b>(i+l, j+k)[2];
                  n_match++;
                  found_match = true;
                }
              }
            }
            k_size++;
          }
          bool_image.at<uchar>(i, j) = 1;
          if (n_match > (float)(((k_size + 1)*(k_size + 1) - 1))/2.0) {
            new_image.at<cv::Vec3b>(i, j)[0] = (int)((float)match_b / (float)n_match);
            new_image.at<cv::Vec3b>(i, j)[1] = (int)((float)match_g / (float)n_match);
            new_image.at<cv::Vec3b>(i, j)[2] = (int)((float)match_r / (float)n_match);
          }
        }
      }
    }

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
        }
      }
    }

    out_pointcloud = *new_pcl;
    cv::imshow(CVParams::WINDOW_NAME, out_image_rgb);
    break;
  }
  case 3:
  {
    cv::Mat new_image(in_image_rgb.rows, in_image_rgb.cols, in_image_rgb.type());
    new_image = in_image_rgb.clone();

    std::vector<cv::Point2f> p1;

    cv::Mat frame_gray;
    cv::cvtColor(in_image_rgb, frame_gray, cv::COLOR_BGR2GRAY);

    // calculate optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::TermCriteria criteria;
    try {
      criteria = cv::TermCriteria((cv::TermCriteria::COUNT) +(cv::TermCriteria::EPS), 10, 0.03);
      cv::calcOpticalFlowPyrLK(CVParams::old_3_gray, frame_gray, CVParams::points_3_old, p1, status, err, cv::Size(15, 15), 2, criteria);
    } catch(const std::exception& e){
      std::cerr << e.what() << '\n';
      break;
    }
    

    std::vector<cv::Point2f> good_new;
    float aprox_speed_x = 0;
    float aprox_speed_y = 0;
    int n_points = 0;
    float X_old, Y_old;
    float X, Y, Z;
    for (uint i = 0; i < CVParams::points_3_old.size(); i++) {
      // Select good points
      if (status[i] == 1) {
        good_new.push_back(p1[i]);
        // draw the tracks
        cv::line(new_image, p1[i], CVParams::points_3_old[i], cv::Scalar(0,0,255), 2);
        Z = in_image_depth.at<float>(p1[i].y, p1[i].x);
        if (std::isinf(Z) || std::isnan(Z)) {
          continue;
        } else {
          if (!CVParams::is_depth_in_meters) {
            Z /= 1000;
          }
          X = ((float(p1[i].x) - cx) * Z) / fx;
          Y = ((float(p1[i].y) - cy) * Z) / fy;
          X_old = ((float(CVParams::points_3_old[i].x) - cx) * Z) / fx;
          Y_old = ((float(CVParams::points_3_old[i].y) - cy) * Z) / fy;
        }
        
        if (std::abs(X - X_old) < 1000 && std::abs(Y - Y_old) < 1000 && (std::abs(X - X_old) > 0.01 || std::abs(Y - Y_old) > 0.01)) {
          // TODO: change this to a vector ordered by distance
          aprox_speed_x += X - X_old;
          aprox_speed_y += Y - Y_old;
          n_points++;
        }

      }
    }

    // TODO: change this to only be of the ones in the middle
    aprox_speed_x /= n_points;
    aprox_speed_y /= n_points;
    
    struct timeval time_now {};
    gettimeofday(&time_now, nullptr);
    time_t msecs_time_now = (time_now.tv_sec * 1000) + (time_now.tv_usec / 1000);
    time_t msecs_time_old = (CVParams::last_time.tv_sec * 1000) + (CVParams::last_time.tv_usec / 1000);

    aprox_speed_x = aprox_speed_x / (float(msecs_time_now - msecs_time_old) / 1000);
    aprox_speed_y = aprox_speed_y / (float(msecs_time_now - msecs_time_old) / 1000);

    float speed_mod = std::sqrt(aprox_speed_x*aprox_speed_x+aprox_speed_y*aprox_speed_y);

    if (n_points == 0) {
      speed_mod = 0;
      aprox_speed_x = 0;
      aprox_speed_y = 0;
    }

    // TODO: clean this mess
    cv::Point2f center = cv::Point2f(in_image_rgb.cols/2,in_image_rgb.rows/2);
    if (aprox_speed_x > 0.4) {
      if (aprox_speed_y > 0.4) {
        cv::arrowedLine(new_image, center, cv::Point(center.x - 150, center.y - 150), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      } else if (aprox_speed_y < -0.4) {
        cv::arrowedLine(new_image, center, cv::Point(center.x - 150, center.y + 150), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      } else {
        cv::arrowedLine(new_image, center, cv::Point(center.x - 150, center.y), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      }
    } else if (aprox_speed_x < -0.4) {
      if (aprox_speed_y > 0.4) {
        cv::arrowedLine(new_image, center, cv::Point(center.x + 150, center.y - 150), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      } else if (aprox_speed_y < -0.4) {
        cv::arrowedLine(new_image, center, cv::Point(center.x + 150, center.y + 150), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      } else {
        cv::arrowedLine(new_image, center, cv::Point(center.x + 150, center.y), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      };
    } else {
      if (aprox_speed_y > 0.4) {
        cv::arrowedLine(new_image, center, cv::Point(center.x, center.y - 150), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      } else if (aprox_speed_y < -0.4) {
        cv::arrowedLine(new_image, center, cv::Point(center.x, center.y + 150), cv::Scalar(0, 0, 0), 10, 8, 0, 0.3);
      }
    }


    cv::rectangle(new_image, cv::Point(0,0), cv::Point(155,55), cv::Scalar(255,255,255), -1);
    cv::rectangle(new_image, cv::Point(0,0), cv::Point(155,55), cv::Scalar(0,0,0), 1);

    printf("[%.2f, %.2f] Speed Mod: %.2f\n", aprox_speed_x, aprox_speed_y, speed_mod);

    std::string text = "Total speed: " + fmt::format("{:.2f}", speed_mod);
    cv::putText(new_image, text, cv::Point(10,15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    text = " ~ X: " + fmt::format("{:.2f}", aprox_speed_x);
    cv::putText(new_image, text, cv::Point(10,35), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    text = " ~ Y: " + fmt::format("{:.2f}", aprox_speed_y);
    cv::putText(new_image, text, cv::Point(10,50), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));


    // Now update the previous frame, previous points and time
    gettimeofday(&CVParams::last_time, nullptr);
    CVParams::old_3_gray = frame_gray.clone();
    CVParams::points_3_old = good_new;

    // Take first frame and find corners in it only if moved
    if (std::abs(aprox_speed_x) > 0.3 || std::abs(aprox_speed_y) > 0.3) {
      cv::cvtColor(in_image_rgb, CVParams::old_3_gray, cv::COLOR_BGR2GRAY);
      cv::goodFeaturesToTrack(CVParams::old_3_gray, CVParams::points_3_old, 100, 0.3, 7, cv::Mat(), 7, false, 0.04);
    }

    cv::imshow(CVParams::WINDOW_NAME, new_image);
    break;
  }
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
