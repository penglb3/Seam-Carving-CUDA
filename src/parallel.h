#ifndef _PARALLEL_H_
#define _PARALLEL_H_

#include "project.h"

/*  TODO: 
    Optimize the memory allocs and frees in the following functions.
    You can see that they do not reuse the memory between functions, inducing quite some overhead.
    If we reuse memory buffers between functions, we should be able to get some more performance.
*/
void getEnergyMap(cv::Mat& h_energy, cv::Mat& h_energyMap, int rowSize, int colSize);
int getMinCumulativeEnergy(cv::Mat& h_energyMap, int rowSize, int colSize);
void removeSeam(cv::Mat& h_image, std::vector<int> h_seam);
void wrapper(cv::Mat& image, int& reduceWidth, int& reduceHeight);

#endif
