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
        sobelEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();

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
        cumEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();

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
        findSeamTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
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
        removeSeamTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        return;
    }

    vector<int> FordFulkersonFindSeam(Mat& energy) {
        int *edge_from, *edge_to;
        double *edge_capacity, *edge_flow;
        int *neighbor, *direction, *residual, *state;

        int n = energy.rows;
        int m = energy.cols;

        // 0 for source, 1 for sink
        int node_size = 2 + n * (m + 1);
        neighbor = new int[node_size];
        direction = new int[node_size];
        residual = new int[node_size];
        state = new int[node_size];

        for (int i = 0; i < node_size; i++) {
            neighbor[i] = 0;
            direction[i] = 0;
            residual[i] = 0;
            state[i] = UNLABELED;
        }

        int edge_size = 2 * n + 2 * n * m + 2  * (n - 1) * m;
        edge_from = new int[edge_size];
        edge_to = new int[edge_size];
        edge_capacity = new double[edge_size];
        edge_flow = new double[edge_size];

        int edge_cnt = 0;
        // source and sink related edges
        for (int i = 0; i < n; i++) {
            // source
            edge_from[edge_cnt] = 0;
            edge_to[edge_cnt] = i * (m + 1) + 2;
            edge_capacity[edge_cnt] = numeric_limits<double>::max() / 2.0;
            edge_flow[edge_cnt] = 0.0;
            edge_cnt++;
            // sink
            edge_from[edge_cnt] = i * (m + 1) + m + 2;
            edge_to[edge_cnt] = 1;
            edge_capacity[edge_cnt] = numeric_limits<double>::max() / 2.0;
            edge_flow[edge_cnt] = 0.0;
            edge_cnt++;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m + 1; j++) {
                if (j < m){
                    edge_from[edge_cnt] = i * (m + 1) + j + 2;
                    edge_to[edge_cnt] = i * (m + 1) + j + 1 + 2;
                    edge_capacity[edge_cnt] = energy.at<double>(i, j);
                    edge_flow[edge_cnt] = 0.0;
                    edge_cnt++;
                }
                if (j > 0) {
                    edge_from[edge_cnt] = i * (m + 1) + j + 2;
                    edge_to[edge_cnt] = i * (m + 1) + (j - 1) + 2;
                    edge_capacity[edge_cnt] = numeric_limits<double>::max() / 2.0;
                    edge_flow[edge_cnt] = 0.0;
                    edge_cnt++;
                }
                if (j > 0 && i < n - 1) {
                    edge_from[edge_cnt] = i * (m + 1) + j + 2;
                    edge_to[edge_cnt] = (i + 1) * (m + 1) + (j - 1) + 2;
                    edge_capacity[edge_cnt] = numeric_limits<double>::max() / 2.0;
                    edge_flow[edge_cnt] = 0.0;
                    edge_cnt++;
                }
                if (j > 0 && i > 0) {
                    edge_from[edge_cnt] = i * (m + 1) + j + 2;
                    edge_to[edge_cnt] = (i - 1) * (m + 1) + (j - 1) + 2;
                    edge_capacity[edge_cnt] = numeric_limits<double>::max() / 2.0;
                    edge_flow[edge_cnt] = 0.0;
                    edge_cnt++;
                }
            }
        }
        assert(edge_cnt == edge_size);
    }
}