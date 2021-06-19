#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "project.h"

using namespace std;
using namespace cv;

constexpr int UNLABELED = 0;
constexpr int UNSCANNED = 1;
constexpr int SCANNED = 2;

constexpr int IN = 1;
constexpr int OUT = 0;

extern float sobelEnergyTime;
extern float cumEnergyTime;
extern float findSeamTime;
extern float removeSeamTime;
extern float transposeTime;
extern float totalTime;

extern int visualize;
extern double* time_records;
extern int records_idx;

namespace CPU{
    void trans(Mat& image){
        transpose(image, image);
    }

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
        grad.convertTo(energy, CV_64F, 1.0 / 255.0);

        auto end = chrono::high_resolution_clock::now();
        sobelEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;

        return energy;
    }

    Mat createEnergyMap(Mat& energy) {
        auto start = chrono::high_resolution_clock::now();
        double topCenter, topLeft, topRight;
        int rowSize = energy.rows;
        int colSize = energy.cols;

        // Initialize energy map
        Mat energyMap = Mat(rowSize, colSize, CV_64F, double(0));

        // Vertical Seam
        // The first row of the map should be the same as the first row of energy
        energy.row(0).copyTo(energyMap.row(0));

        for (int row = 1; row < rowSize; row++) {
            for (int col = 0; col < colSize; col++) {
                topCenter = energyMap.at<double>(row - 1, col);
                topLeft = energyMap.at<double>(row - 1, max(col - 1, 0));
                topRight = energyMap.at<double>(row - 1, min(col + 1, colSize - 1));

                // add energy at pixel with smallest of previous row neighbor's cumulative energy
                energyMap.at<double>(row, col) = energy.at<double>(row, col) + min(topCenter, min(topLeft, topRight));
            }
        }

        auto end = chrono::high_resolution_clock::now();
        cumEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;

        return energyMap;
    }

    vector<int> findSeam(Mat& energyMap) {
        auto start = chrono::high_resolution_clock::now();
        int rowSize = energyMap.rows;
        int colSize = energyMap.cols;
        vector<int> seam;
        double topCenter, topLeft, topRight;
        double minVal;
        Point minLoc;
        
        // Vertical seam, reduces width
        seam.resize(rowSize);
        // Get location of min cumulative energy
        minMaxLoc(energyMap.row(rowSize - 1), &minVal, NULL, &minLoc, NULL);
        int curCol = minLoc.x;
        seam[rowSize - 1] = curCol;

        // Look at top neighbors to find next minimum cumulative energy
        for (int row = rowSize - 1; row > 0; row--) {
            topCenter = energyMap.at<double>(row - 1, curCol);
            topLeft = energyMap.at<double>(row - 1, max(curCol - 1, 0));
            topRight = energyMap.at<double>(row - 1, min(curCol + 1, colSize - 1));

            // find next col idx
            if (min(topLeft, topCenter) > topRight) {
                // topRight smallest
                curCol += 1;
            }
            else if (min(topRight, topCenter) > topLeft) {
                // topLeft smallest
                curCol -= 1;
            }
            // if topCenter smallest, curCol remain;
            // update seam
            seam[row - 1] = curCol;
        }

        auto end = chrono::high_resolution_clock::now();
        findSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return seam;
    }

    void removeSeam(Mat& image, vector<int> seam) {
        auto start = chrono::high_resolution_clock::now();
        // spare 1x1x3 to maintain matrix size
        Mat spare(1, 1, CV_8UC3, Vec3b(0, 0, 0));

        // Vertical seam, reduces width
        Mat tempRow(image.cols, 1, CV_8UC3);
        for (int i = 0; i < image.rows; i++) {
            tempRow.setTo(0);
            Mat beforeIdx = image.rowRange(i, i + 1).colRange(0, seam[i]);
            Mat afterIdx = image.rowRange(i, i + 1).colRange(seam[i] + 1, image.cols);

            if (beforeIdx.empty()) {
                hconcat(afterIdx, spare, tempRow);
            }
            else if (afterIdx.empty()) {
                hconcat(beforeIdx, spare, tempRow);
            }
            else {
                hconcat(beforeIdx, afterIdx, tempRow);
                hconcat(tempRow, spare, tempRow);
            }
            tempRow.copyTo(image.row(i));
        }
        image = image.colRange(0, image.cols - 1);
        
        //imshow("after cut", image);
        auto end = chrono::high_resolution_clock::now();
        removeSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return;
    }

    void wrapper(Mat& image, int& reduceWidth, int& reduceHeight)
    {
        Mat energy, energyMap, h_temp;
        vector<int> seam;
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point record;
        if (visualize == 1)
        {
            record = std::chrono::high_resolution_clock::now();
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
        }
        // Vertical seam
        for (int i = 0; i < reduceWidth; i++) {
            energy = createEnergyImg(image);
            energy = createEnergyMap(energy);
            seam = findSeam(energy);
            removeSeam(image, seam);
            if (visualize == 1)
            {
                record = std::chrono::high_resolution_clock::now();
                time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
            }
        }
        auto startTranspose = std::chrono::high_resolution_clock::now();
        trans(image);
        auto endTranspose = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        if (visualize == 1)
        {
            record = std::chrono::high_resolution_clock::now();
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
        }
        // Horizontal seam
        for (int j = 0; j < reduceHeight; j++) {
            energy = createEnergyImg(image);
            seam = findSeam(energy);
            removeSeam(image, seam);
            if (visualize == 1)
            {
                record = std::chrono::high_resolution_clock::now();
                time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
            }
        }
        startTranspose = std::chrono::high_resolution_clock::now();
        trans(image);
        auto end = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - startTranspose).count() / 1e3;
        totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
        if (visualize == 1)
        {
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
        }
    }
}
