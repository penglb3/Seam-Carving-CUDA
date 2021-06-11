#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>

#include "project.h"
#include "getopt.hpp"

using namespace cv;
using std::string;
using std::cout;
using std::pair;
using std::endl;
using std::vector;

float sobelEnergyTime = 0;
float cumEnergyTime = 0;
float findSeamTime = 0;
float removeSeamTime = 0;

int main(int argc, char** argv)
{
	auto start = std::chrono::high_resolution_clock::now();
	// Set how much to reduce width or/and height by and set image.
	int reduceWidth = 300;
	int reduceHeight = 0;
	string imageName = "../images/Tension.jpg";
	Mat image = imread(imageName, IMREAD_COLOR);
	if (image.empty()) {
		cout << "Invalid image. Please try again" << endl;
		waitKey(0);
		return 1;
	}
	pair<int, int> imageSize = { image.cols, image.rows };

	imshow("Original", image);

    Mat (*createEnergyImg)(Mat &) = CUDA::createEnergyImg;
    Mat (*createEnergyMap)(Mat&, eSeamDirection) = CUDA::createEnergyMap;
    vector<int> (*findSeam)(Mat&, eSeamDirection) = CUDA::findSeam;
    void (*removeSeam)(Mat&, vector<int>, eSeamDirection) = CUDA::removeSeam;

	// Vertical seam, reduces width
	for (int i = 0; i < reduceWidth; i++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, VERTICAL);
		vector<int> seam = findSeam(energyMap, VERTICAL);
		removeSeam(image, seam, VERTICAL);
	}

	// Horizontal seam, reduces height
	for (int j = 0; j < reduceHeight; j++) {
		Mat energy = createEnergyImg(image);
		Mat energyMap = createEnergyMap(energy, HORIZONTAL);
		vector<int> seam = findSeam(energyMap, HORIZONTAL);
		removeSeam(image, seam, HORIZONTAL);
	}

	imshow("Result", image);
	auto end = std::chrono::high_resolution_clock::now();
	float totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	cout << "Sequential on CPU" << endl;
	cout << "Image name: " << imageName << endl;
	cout << "Input dimension " << imageSize.first << " x " << imageSize.second << endl;
	cout << "Output dimension " << image.cols << " x " << image.rows << endl << endl;
	cout << "Time taken to get energy of each image: " << sobelEnergyTime << "(ms)" << endl;
	cout << "Time taken to get cumulative energy map: " << cumEnergyTime << "(ms)" << endl;
	cout << "Time taken to find seam: " << findSeamTime << "(ms)" << endl;
	cout << "Time taken to remove seam: " << removeSeamTime << "(ms)" << endl;
	cout << "Total time: " << totalTime << "(ms)" << endl;

	imwrite("../images/output.jpg", image);
	waitKey(0);

	system("pause");
	return 0;
}