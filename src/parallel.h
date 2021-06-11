#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "project.h"

void warmUpGPU();
void getEnergyMap(cv::Mat& h_energy, cv::Mat& h_energyMap, int rowSize, int colSize);
int getMinCumulativeEnergy(cv::Mat& h_energyMap, int rowSize, int colSize);
void removeSeam(cv::Mat& h_image, std::vector<int> h_seam);

#endif