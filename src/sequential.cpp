#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include "project.h"

using namespace std;
using namespace cv;

namespace CPU{
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

	Mat createEnergyMap(Mat& energy, eSeamDirection seamDirection) {
		auto start = chrono::high_resolution_clock::now();
		double topCenter, topLeft, topRight;
		int rowSize = energy.rows;
		int colSize = energy.cols;

		// Initialize energy map
		Mat energyMap = Mat(rowSize, colSize, CV_64F, double(0));

		if (seamDirection == VERTICAL) {
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
		}
		else {
			// Horizontal Seam
			// The first col of the map should be the same as the first col of energy
			energy.col(0).copyTo(energyMap.col(0));

			for (int col = 1; col < colSize; col++) {
				for (int row = 0; row < rowSize; row++) {	
					topCenter = energyMap.at<double>(row, col - 1);
					topLeft = energyMap.at<double>(max(row - 1, 0), col - 1);
					topRight = energyMap.at<double>(min(row + 1, rowSize - 1), col - 1);

					// add energy at pixel with smallest of previous col neighbor's cumulative energy
					energyMap.at<double>(row, col) = energy.at<double>(row, col) + min(topCenter, min(topLeft, topRight));
				}
			}
		}
		auto end = chrono::high_resolution_clock::now();
		cumEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();

		return energyMap;
	}

	vector<int> findSeam(Mat& energyMap, eSeamDirection seamDirection) {
		auto start = chrono::high_resolution_clock::now();
		int rowSize = energyMap.rows;
		int colSize = energyMap.cols;
		vector<int> seam;
		double topCenter, topLeft, topRight;
		double minVal;
		Point minLoc;
		
		if (seamDirection == VERTICAL) {
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
		}
		else {
			// Horizontal seam, reduces height
			seam.resize(colSize);
			// Get location of min cumulative energy
			minMaxLoc(energyMap.col(colSize - 1), &minVal, NULL, &minLoc, NULL);
			int curCol = minLoc.y;
			seam[colSize - 1] = curCol;

			// Look at top neighbors to find next minimum cumulative energy
			for (int col = colSize - 1; col > 0; col--) {
				topCenter = energyMap.at<double>(curCol, col - 1);
				topLeft = energyMap.at<double>(max(curCol - 1, 0), col - 1);
				topRight = energyMap.at<double>(min(curCol + 1, rowSize - 1), col - 1);

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
				seam[col - 1] = curCol;
			}
		}
		auto end = chrono::high_resolution_clock::now();
		findSeamTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
		return seam;
	}

	void removeSeam(Mat& image, vector<int> seam, eSeamDirection seamDirection) {
		auto start = chrono::high_resolution_clock::now();
		// spare 1x1x3 to maintain matrix size
		Mat spare(1, 1, CV_8UC3, Vec3b(0, 0, 0));

		if (seamDirection == VERTICAL) {
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
		}
		else {
			// Horizontal seam, reduces height
			Mat tempCol(1, image.rows, CV_8UC3);
			for (int i = 0; i < image.cols; i++) {
				tempCol.setTo(0);
				Mat beforeIdx = image.colRange(i, i + 1).rowRange(0, seam[i]);
				Mat afterIdx = image.colRange(i, i + 1).rowRange(seam[i] + 1, image.rows);

				if (beforeIdx.empty()) {
					vconcat(afterIdx, spare, tempCol);
				}
				else if (afterIdx.empty()) {
					vconcat(beforeIdx, spare, tempCol);
				}
				else {
					vconcat(beforeIdx, afterIdx, tempCol);
					vconcat(tempCol, spare, tempCol);
				}
				tempCol.copyTo(image.col(i));
			}
			image = image.rowRange(0, image.rows - 1);
		}
		
		//imshow("after cut", image);
		auto end = chrono::high_resolution_clock::now();
		removeSeamTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
		return;
	}
}