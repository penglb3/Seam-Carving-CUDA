#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "project.h"

void getEnergyMap(cv::cuda::GpuMat& d_energy, cv::cuda::GpuMat& d_energyMap, int rows, int cols);
int getMinCumulativeEnergy(cv::cuda::GpuMat& d_energyMap);
void removeSeam(cv::cuda::GpuMat& d_image, std::vector<int> h_seam);

class MyAllocator : public cv::cuda::GpuMat::Allocator
{
public:
    bool allocate(cv::cuda::GpuMat* mat, int rows, int cols, size_t elemSize) CV_OVERRIDE;
    void free(cv::cuda::GpuMat* mat) CV_OVERRIDE;
};

#endif
