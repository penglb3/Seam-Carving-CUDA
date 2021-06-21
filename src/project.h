#ifndef _PROJECT_H_
#define _PROJECT_H_

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/videoio.hpp>

namespace CPU{
    cv::Mat createEnergyImg(cv::Mat &image);
    cv::Mat createEnergyMap(cv::Mat& energy);
    std::vector<int> findSeam(cv::Mat& energyMap);
    void removeSeam(cv::Mat& image, std::vector<int> seam);
    void trans(cv::Mat& image);
    void wrapper(cv::Mat& image, int& reduceWidth, int& reduceHeight);
}

namespace CUDA{
    void warmUpGPU();
    cv::cuda::GpuMat createEnergyImg(cv::cuda::GpuMat& image);
    cv::cuda::GpuMat createEnergyMap(cv::cuda::GpuMat& d_energy);
    std::vector<int> findSeam(cv::cuda::GpuMat& d_energyMap);
    void removeSeam(cv::cuda::GpuMat& d_image, std::vector<int> seam);
    void trans(cv::cuda::GpuMat& d_image);
    void wrapper(cv::Mat& image, int& reduceWidth, int& reduceHeight);
}

extern float sobelEnergyTime;
extern float cumEnergyTime;
extern float findSeamTime;
extern float removeSeamTime;

#endif
