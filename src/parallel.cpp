#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "parallel.h"

using namespace std;
using namespace cv;
using namespace cv::cuda;

extern float sobelEnergyTime;
extern float cumEnergyTime;
extern float findSeamTime;
extern float removeSeamTime;
extern float transposeTime;
extern float totalTime;

MyAllocator myAllocator;

namespace CUDA{
    // Mat createEnergyMap(Mat& energy) {
    GpuMat createEnergyMap(GpuMat& d_energy) {
        auto start = chrono::high_resolution_clock::now();
        // int rowSize = energy.rows;
        int rowSize = d_energy.rows;
        // int colSize = energy.cols;
        int colSize = d_energy.cols;
        // Initialize energy map
        // Mat energyMap = Mat(rowSize, colSize, CV_32F, float(0));
        GpuMat d_energyMap(rowSize, colSize, CV_32F, float(0));

        // Call cuda function to get energy map
        // getEnergyMap(energy, energyMap, rowSize, colSize);
        getEnergyMap(d_energy, d_energyMap, rowSize, colSize);

        auto end = chrono::high_resolution_clock::now();
        cumEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        // return energyMap;
        return d_energyMap;
    }

    // vector<int> findSeam(Mat& energyMap) {
    vector<int> findSeam(GpuMat& d_energyMap) {
        auto start = chrono::high_resolution_clock::now();
        // int rowSize = energyMap.rows;
        int rowSize = d_energyMap.rows;
        // int colSize = energyMap.cols;
        int colSize = d_energyMap.cols;
        int curLoc;
        vector<int> seam;
        float topCenter, topLeft, topRight;
        double minVal;
        Point minLoc;

        seam.resize(rowSize);

        // curLoc = getMinCumulativeEnergy(energyMap, rowSize, colSize);
        curLoc = getMinCumulativeEnergy(d_energyMap, rowSize, colSize);

        Mat h_energyMap;
        d_energyMap.download(h_energyMap);
        d_energyMap.release();

        seam[rowSize - 1] = curLoc;

        // Look at top neighbors to find next minimum cumulative energy
        for (int row = rowSize - 1; row > 0; row--) {
            // topCenter = energyMap.at<float>(row - 1, curLoc);
            topCenter = h_energyMap.at<float>(row - 1, curLoc);
            // topLeft = energyMap.at<float>(row - 1, max(curLoc - 1, 0));
            topLeft = h_energyMap.at<float>(row - 1, max(curLoc - 1, 0));
            // topRight = energyMap.at<float>(row - 1, min(curLoc + 1, colSize - 1));
            topRight = h_energyMap.at<float>(row - 1, min(curLoc + 1, colSize - 1));

            // find next col idx
            if (min(topLeft, topCenter) > topRight) {
                // topRight smallest
                curLoc += 1;
            }
            else if (min(topRight, topCenter) > topLeft) {
                // topLeft smallest
                curLoc -= 1;
            }
            // if topCenter smallest, curCol remain;
            // update seam
            seam[row - 1] = curLoc;
        }

        h_energyMap.release();
        
        auto end = chrono::high_resolution_clock::now();
        findSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return seam;
    }

    void wrapper(Mat& image, int& reduceWidth, int& reduceHeight)
    {
        GpuMat::setDefaultAllocator(&myAllocator);
        GpuMat d_image, d_energy, d_energyMap;
        // Mat energy, energyMap;
        vector<int> seam;
        auto start = std::chrono::high_resolution_clock::now();
        // Vertical seam
        d_image.upload(image);
        for (int i = 0; i < reduceWidth; i++) {
            // energy = createEnergyImg(image);
            d_energy = createEnergyImg(d_image);
            // energy = createEnergyMap(energy);
            d_energyMap = createEnergyMap(d_energy);
            d_energy.release();
            // seam = findSeam(energy);
            seam = findSeam(d_energyMap);
            d_energyMap.release();
            // removeSeam(image, seam);
            removeSeam(d_image, seam);
        }
        auto startTranspose = std::chrono::high_resolution_clock::now();
        // trans(image);
        trans(d_image);
        auto endTranspose = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        // Horizontal seam
        for (int j = 0; j < reduceHeight; j++) {
            // energy = createEnergyImg(image);
            d_energy = createEnergyImg(d_image);
            // energy = createEnergyMap(energy);
            d_energyMap = createEnergyMap(d_energy);
            d_energy.release();
            // seam = findSeam(energy);
            seam = findSeam(d_energyMap);
            d_energyMap.release();
            // removeSeam(image, seam);
            removeSeam(d_image, seam);
        }
        startTranspose = std::chrono::high_resolution_clock::now();
        // trans(image);
        trans(d_image);
        endTranspose = std::chrono::high_resolution_clock::now();
        // transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - startTranspose).count() / 1e3;
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        d_image.download(image);
        d_image.release();
        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    }
}
