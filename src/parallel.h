#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "project.h"

void getEnergyMap(cv::cuda::GpuMat& d_energy, cv::cuda::GpuMat& d_energyMap, int rowSize, int colSize);
int getMinCumulativeEnergy(cv::cuda::GpuMat& d_energyMap, int rowSize, int colSize);
void removeSeam(cv::cuda::GpuMat& d_image, std::vector<int> h_seam);
void wrapper(cv::Mat& image, int& reduceWidth, int& reduceHeight);

class MyAllocator : public cv::cuda::GpuMat::Allocator
{
public:
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE;
    void free(cv::cuda::GpuMat* mat) CV_OVERRIDE;
};

#endif
