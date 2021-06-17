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

namespace CUDA{
    Mat createEnergyMap(Mat& energy) {
        auto start = chrono::high_resolution_clock::now();
        int rowSize = energy.rows;
        int colSize = energy.cols;
        // Initialize energy map
        Mat energyMap = Mat(rowSize, colSize, CV_32F, float(0));

        // Call cuda function to get energy map
        getEnergyMap(energy, energyMap, rowSize, colSize);

        auto end = chrono::high_resolution_clock::now();
        cumEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return energyMap;
    }

    vector<int> findSeam(Mat& energyMap) {
        auto start = chrono::high_resolution_clock::now();
        int rowSize = energyMap.rows;
        int colSize = energyMap.cols;
        int curLoc;
        vector<int> seam;
        float topCenter, topLeft, topRight;
        double minVal;
        Point minLoc;

        seam.resize(rowSize);

        curLoc = getMinCumulativeEnergy(energyMap, rowSize, colSize);

        seam[rowSize - 1] = curLoc;

        // Look at top neighbors to find next minimum cumulative energy
        for (int row = rowSize - 1; row > 0; row--) {
            topCenter = energyMap.at<float>(row - 1, curLoc);
            topLeft = energyMap.at<float>(row - 1, max(curLoc - 1, 0));
            topRight = energyMap.at<float>(row - 1, min(curLoc + 1, colSize - 1));

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
        
        auto end = chrono::high_resolution_clock::now();
        findSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return seam;
    }

    void wrapper(Mat& image, int& reduceWidth, int& reduceHeight)
    {
        Mat energy, energyMap;
        vector<int> seam;
        auto start = std::chrono::high_resolution_clock::now();
        // Vertical seam
        for (int i = 0; i < reduceWidth; i++) {
            energy = createEnergyImg(image);
            energy = createEnergyMap(energy);
            seam = findSeam(energy);
            removeSeam(image, seam);
        }
        auto startTranspose = std::chrono::high_resolution_clock::now();
        trans(image);
        auto endTranspose = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        // Horizontal seam
        for (int j = 0; j < reduceHeight; j++) {
            energy = createEnergyImg(image);
            seam = findSeam(energy);
            removeSeam(image, seam);
        }
        startTranspose = std::chrono::high_resolution_clock::now();
        trans(image);
        auto end = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - startTranspose).count() / 1e3;
        totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
    }
}
