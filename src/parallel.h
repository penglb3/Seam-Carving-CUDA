#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "project.h"

void warmUpGPU();
void getEnergyMap(cv::Mat& h_energy, cv::Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection);
int getMinCumulativeEnergy(cv::Mat& h_energyMap, int rowSize, int colSize, eSeamDirection seamDirection);
void removeSeam(cv::Mat& h_image, std::vector<int> h_seam, eSeamDirection seamDirection);

#endif