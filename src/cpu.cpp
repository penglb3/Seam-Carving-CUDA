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

    Mat calculateEnergyImg(Mat &image) {
        auto start = chrono::high_resolution_clock::now();
        Mat grayscale, grad, energy;
        Mat grad_x, grad_y;
        Mat abs_grad_x, abs_grad_y;
        int ddepth = CV_16S;
        int scale = 1;
        int delta = 0;

        cvtColor(image, grayscale, COLOR_BGR2GRAY);

        Sobel(grayscale, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
        Sobel(grayscale, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);

        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

        grad.convertTo(energy, CV_64F, 1.0 / 255.0);

        auto end = chrono::high_resolution_clock::now();
        sobelEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;

        return energy;
    }

    Mat calculateEnergyMap(Mat& energy) {
        auto start = chrono::high_resolution_clock::now();
        double upper, upperLeft, upperRight;
        int rows = energy.rows;
        int cols = energy.cols;

        Mat energyMap = Mat(rows, cols, CV_64F, double(0));

        energy.row(0).copyTo(energyMap.row(0));

        for (int row = 1; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                upper = energyMap.at<double>(row - 1, col);
                upperLeft = energyMap.at<double>(row - 1, max(col - 1, 0));
                upperRight = energyMap.at<double>(row - 1, min(col + 1, cols - 1));

                energyMap.at<double>(row, col) = energy.at<double>(row, col) + min(upper, min(upperLeft, upperRight));
            }
        }

        auto end = chrono::high_resolution_clock::now();
        cumEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;

        return energyMap;
    }

    vector<int> findSeam(Mat& energyMap) {
        auto start = chrono::high_resolution_clock::now();
        int rows = energyMap.rows;
        int cols = energyMap.cols;
        vector<int> seam;
        float upper, upper_left, upper_right;
        double minVal;
        Point minLoc;

        seam.resize(rows);

        minMaxLoc(energyMap.row(rows - 1), &minVal, NULL, &minLoc, NULL);
        int current = minLoc.x;
        seam[rows - 1] = current;

        for (int row = rows - 2; row >= 0; row--) {
            upper = energyMap.at<float>(row, current);
            upper_left = energyMap.at<float>(row, max(current - 1, 0));
            upper_right = energyMap.at<float>(row, min(current + 1, cols - 1));

            // find next col idx
            if (min(upper_left, upper) > upper_right) {
                // upper_right smallest
                current += 1;
            }
            else if (min(upper_right, upper) > upper_left) {
                // upper_left smallest
                current -= 1;
            }
            seam[row] = current;
        }

        auto end = chrono::high_resolution_clock::now();
        findSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return seam;
    }

    void removeSeam(Mat& image, vector<int> seam) {
        auto start = chrono::high_resolution_clock::now();
        Mat spare(1, 1, CV_8UC3, Vec3b(0, 0, 0));
        Mat temp(image.cols, 1, CV_8UC3);
        for (int i = 0; i < image.rows; i++) {
            temp.setTo(0);
            Mat beforeIdx = image.rowRange(i, i + 1).colRange(0, seam[i]);
            Mat afterIdx = image.rowRange(i, i + 1).colRange(seam[i] + 1, image.cols);

            if (beforeIdx.empty()) {
                hconcat(afterIdx, spare, temp);
            }
            else if (afterIdx.empty()) {
                hconcat(beforeIdx, spare, temp);
            }
            else {
                hconcat(beforeIdx, afterIdx, temp);
                hconcat(temp, spare, temp);
            }
            temp.copyTo(image.row(i));
        }
        image = image.colRange(0, image.cols - 1);
        
        auto end = chrono::high_resolution_clock::now();
        removeSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return;
    }

    void seamCarve(Mat& image, int& reduceWidth, int& reduceHeight) {
        Mat energy, energyMap, h_temp;
        vector<int> seam;
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point record;
        if (visualize == 2) {
            record = std::chrono::high_resolution_clock::now();
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
        }
        // Vertical seam
        for (int i = 0; i < reduceWidth; i++) {
            energy = calculateEnergyImg(image);
            energy = calculateEnergyMap(energy);
            seam = findSeam(energy);
            removeSeam(image, seam);
            if (visualize == 2) {
                record = std::chrono::high_resolution_clock::now();
                time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
            }
        }
        auto startTranspose = std::chrono::high_resolution_clock::now();
        trans(image);
        auto endTranspose = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        if (visualize == 2) {
            record = std::chrono::high_resolution_clock::now();
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
        }
        // Horizontal seam
        for (int j = 0; j < reduceHeight; j++) {
            energy = calculateEnergyImg(image);
            seam = findSeam(energy);
            removeSeam(image, seam);
            if (visualize == 2) {
                record = std::chrono::high_resolution_clock::now();
                time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
            }
        }
        startTranspose = std::chrono::high_resolution_clock::now();
        trans(image);
        auto end = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(end - startTranspose).count() / 1e3;
        totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
        if (visualize == 2) {
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
        }
    }
}
