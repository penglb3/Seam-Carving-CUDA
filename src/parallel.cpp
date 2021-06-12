#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "parallel.h"

using namespace std;
using namespace cv;

namespace CUDA{
    Mat createEnergyImg(Mat &image) {
        auto start = chrono::high_resolution_clock::now();
        Mat grayscale, grad, energy;
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        int ddepth = CV_16S;
        int scale = 1;
        int delta = 0;

        // Convert image to grayscale
        cvtColor(image, grayscale, COLOR_BGR2GRAY);

        // Perform sobel operator to get image gradient
        Sobel(grayscale, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        Sobel(grayscale, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        // Convert gradient to double
        grad.convertTo(energy, CV_32F, 1.0 / 255.0);

        auto end = chrono::high_resolution_clock::now();
        sobelEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();

        return energy;
    }

    Mat createEnergyMap(Mat& energy) {
        auto start = chrono::high_resolution_clock::now();
        int rowSize = energy.rows;
        int colSize = energy.cols;
        // Initialize energy map
        Mat energyMap = Mat(rowSize, colSize, CV_32F, float(0));

        // Call cuda function to get energy map
        getEnergyMap(energy, energyMap, rowSize, colSize);

        auto end = chrono::high_resolution_clock::now();
        cumEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
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

        
        // Call kernel for parallel reduction to find min cumulative energy
        seam.resize(rowSize);

        curLoc = getMinCumulativeEnergy(energyMap, rowSize, colSize);
        //cuda::minMaxLoc(energyMap.row(rowSize - 1), &minVal, NULL, &minLoc, NULL);
        //curLoc = minLoc.x;

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
        findSeamTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        return seam;
    }
}