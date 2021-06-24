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

extern int visualize;
extern double fps;
extern double spf;
extern double* time_records;
extern int records_idx;
extern int cur_frame;
extern VideoWriter out_capture;
extern int fW;
extern int fH;

MyAllocator myAllocator;

namespace CUDA{
    GpuMat calculateEnergyMap(GpuMat& d_energy) {
        auto start = chrono::high_resolution_clock::now();
        int rows = d_energy.rows;
        int cols = d_energy.cols;
        // Initialize energy map
        GpuMat d_energyMap(rows, cols, CV_32F, float(0));

        // Call cuda function to get energy map
        getEnergyMap(d_energy, d_energyMap, rows, cols);

        auto end = chrono::high_resolution_clock::now();
        cumEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;

        return d_energyMap;
    }

    vector<int> findSeam(GpuMat& d_energyMap) {
        auto start = chrono::high_resolution_clock::now();
        int rows = d_energyMap.rows;
        int cols = d_energyMap.cols;
        int current;
        vector<int> seam;
        float upper, upper_left, upper_right;
        double minVal;
        Point minLoc;

        seam.resize(rows);

        current = getMinCumulativeEnergy(d_energyMap);

        Mat h_energyMap;
        d_energyMap.download(h_energyMap);
        d_energyMap.release();

        seam[rows - 1] = current;

        // Look at top neighbors to find next minimum cumulative energy
        for (int row = rows - 2; row >= 0; row--) {
            upper = h_energyMap.at<float>(row, current);
            upper_left = h_energyMap.at<float>(row, max(current - 1, 0));
            upper_right = h_energyMap.at<float>(row, min(current + 1, cols - 1));

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

        h_energyMap.release();
        
        auto end = chrono::high_resolution_clock::now();
        findSeamTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;
        return seam;
    }

    void seamCarve(Mat& image, int& reduceWidth, int& reduceHeight) {
        GpuMat::setDefaultAllocator(&myAllocator);
        GpuMat d_image, d_energy, d_energyMap;
        Mat h_temp;
        vector<int> seam;
        auto start = std::chrono::high_resolution_clock::now();
        std::chrono::steady_clock::time_point record;
        // Vertical seam
        d_image.upload(image);
        if (visualize == 1) {
            record = std::chrono::high_resolution_clock::now();
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
        }
        else if (visualize == 2) {
            cv::copyMakeBorder(h_temp, h_temp, 0, fH, 0, fW, h_temp.type());
            double record = time_records[++records_idx];
            while (cur_frame * spf < record) {
                out_capture << h_temp;
                ++cur_frame;
            }
            h_temp.release();
            d_image.download(h_temp);
            cv::copyMakeBorder(h_temp, h_temp, 0, fH-h_temp.rows, 0, fW-h_temp.cols, h_temp.type());
        }
        for (int i = 0; i < reduceWidth; i++) {
            d_energy = calculateEnergyImg(d_image);
            d_energyMap = calculateEnergyMap(d_energy);
            d_energy.release();
            seam = findSeam(d_energyMap);
            d_energyMap.release();
            removeSeam(d_image, seam);
            if (visualize == 1) {
                record = std::chrono::high_resolution_clock::now();
                time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
            }
            else if (visualize == 2) {
                double record = time_records[++records_idx];
                while (cur_frame * spf < record) {
                    out_capture << h_temp;
                    ++cur_frame;
                }
                h_temp.release();
                d_image.download(h_temp);
                cv::copyMakeBorder(h_temp, h_temp, 0, fH-h_temp.rows, 0, fW-h_temp.cols, h_temp.type());
            }
        }
        auto startTranspose = std::chrono::high_resolution_clock::now();
        trans(d_image);
        auto endTranspose = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        if (visualize == 1) {
            record = std::chrono::high_resolution_clock::now();
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
        }
        else if (visualize == 2) {
            double record = time_records[++records_idx];
            while (cur_frame * spf < record) {
                out_capture << h_temp;
                ++cur_frame;
            }
            h_temp.release();
            d_image.download(h_temp);
            cv::copyMakeBorder(h_temp, h_temp, 0, fH-h_temp.rows, 0, fW-h_temp.cols, h_temp.type());
        }
        // Horizontal seam
        for (int j = 0; j < reduceHeight; j++) {
            d_energy = calculateEnergyImg(d_image);
            d_energyMap = calculateEnergyMap(d_energy);
            d_energy.release();
            seam = findSeam(d_energyMap);
            d_energyMap.release();
            removeSeam(d_image, seam);
            if (visualize == 1) {
                record = std::chrono::high_resolution_clock::now();
                time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(record - start).count() / 1e6;
            }
            else if (visualize == 2) {
                double record = time_records[++records_idx];
                while (cur_frame * spf < record) {
                    out_capture << h_temp;
                    ++cur_frame;
                }
                h_temp.release();
                d_image.download(h_temp);
                cv::copyMakeBorder(h_temp, h_temp, 0, fH-h_temp.rows, 0, fW-h_temp.cols, h_temp.type());
            }
        }
        startTranspose = std::chrono::high_resolution_clock::now();
        trans(d_image);
        endTranspose = std::chrono::high_resolution_clock::now();
        transposeTime += std::chrono::duration_cast<std::chrono::microseconds>(endTranspose - startTranspose).count() / 1e3;
        d_image.download(image);
        d_image.release();
        auto end = std::chrono::high_resolution_clock::now();
        totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e3;
        if (visualize == 1) {
            time_records[++records_idx] = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1e6;
        }
        else if (visualize == 2) {
            double record = time_records[++records_idx];
            while (cur_frame * spf < record) {
                out_capture << h_temp;
                ++cur_frame;
            }
            h_temp.release();
            cv::copyMakeBorder(image, h_temp, 0, fH-image.rows, 0, fW-image.cols, image.type());
            out_capture << h_temp;
        }
    }
}
