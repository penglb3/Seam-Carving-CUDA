#ifndef _PROJECT_H_
#define _PROJECT_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>

enum eSeamDirection { VERTICAL, HORIZONTAL };

namespace CPU{
    cv::Mat createEnergyImg(cv::Mat &image);
    cv::Mat createEnergyMap(cv::Mat& energy, eSeamDirection seamDirection);
    std::vector<int> findSeam(cv::Mat& energyMap, eSeamDirection seamDirection);
    void removeSeam(cv::Mat& image, std::vector<int> seam, eSeamDirection seamDirection);
}

namespace CUDA{
    cv::Mat createEnergyImg(cv::Mat &image);
    cv::Mat createEnergyMap(cv::Mat& energy, eSeamDirection seamDirection);
    std::vector<int> findSeam(cv::Mat& energyMap, eSeamDirection seamDirection);
    void removeSeam(cv::Mat& image, std::vector<int> seam, eSeamDirection seamDirection);
}

extern float sobelEnergyTime;
extern float cumEnergyTime;
extern float findSeamTime;
extern float removeSeamTime;

#endif