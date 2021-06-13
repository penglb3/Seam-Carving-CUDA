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
float transposeTime = 0;

int main(int argc, char** argv)
{    
    // Program parameters
    int width = 0, height = 0, parallel = 1,
        blockDimX = 32, blockDimY = 32; // WARNING: blockDim parameters are not really applied.
    
    const char* imageName = "../images/Tension.jpg";
    const char* outputName = "../images/output.jpg";

    // Get parameters from arguments (if provided)
    char c;
    while((c = getopt(argc, argv, "w:h:x:y:i:o:s"))!=-1)
        switch(c){
            case 'x': blockDimX = atoi(optarg);break; 	// block dim x
            case 'y': blockDimY = atoi(optarg);break; 	// block dim y
            case 'w': width = atoi(optarg);break;		// output width
            case 'h': height = atoi(optarg);break;		// output height
            case 'i': imageName = optarg;break;			// input filename
            case 'o': outputName = optarg;break;		// output filename
            case 's': parallel = 0;break;				// use parallel mode?
            default : abort();
        }
    if (!width && !height) {
        cout << "No resizing needed, exiting." << endl;
        system("pause");
        return 0;
    }
    // Set up reduction width and height and read image.
    Mat image = imread(imageName, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Invalid image. Please try again." << endl;
        system("pause");
        return 1;
    }
    pair<int, int> imageSize = { image.cols, image.rows };

    int reduceWidth = (image.cols-width)%image.cols;
    int reduceHeight = (image.rows-height)%image.rows;

    printf(">>>>> Running %s <<<<<\n", parallel?"CUDA":"CPU");
    cout << "Image name: " << imageName << endl;
    cout << "Input dimension " << imageSize.first << " x " << imageSize.second << endl;
    cout << "Output dimension " << image.cols-reduceWidth << " x " << image.rows-reduceHeight << endl << endl;

    // imshow("Original", image);

    // Choose the mode: default = CUDA
    Mat (*createEnergyImg)(Mat&) = CUDA::createEnergyImg;
    Mat (*createEnergyMap)(Mat&) = CUDA::createEnergyMap;
    vector<int> (*findSeam)(Mat&) = CUDA::findSeam;
    void (*removeSeam)(Mat&, vector<int>) = CUDA::removeSeam;
    void (*trans)(Mat&) = CUDA::trans;

    if (!parallel){
        createEnergyImg = CPU::createEnergyImg;
        createEnergyMap = CPU::createEnergyMap;
        findSeam = CPU::findSeam;
        removeSeam = CPU::removeSeam;
        trans = CPU::trans;
    }
    else
        CUDA::warmUpGPU();
    Mat energy, energyMap;
    vector<int> seam;
    auto start = std::chrono::high_resolution_clock::now();
    // Vertical seam
    for (int i = 0; i < reduceWidth; i++) {
        energy = createEnergyImg(image);
        energyMap = createEnergyMap(energy);
        seam = findSeam(energyMap);
        removeSeam(image, seam);
    }
    auto startTranspose = std::chrono::high_resolution_clock::now();
    trans(image);
    auto endTranspose = std::chrono::high_resolution_clock::now();
    transposeTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTranspose - startTranspose).count();
    // Horizontal seam
    for (int j = 0; j < reduceHeight; j++) {
        energy = createEnergyImg(image);
        energyMap = createEnergyMap(energy);
        seam = findSeam(energyMap);
        removeSeam(image, seam);
    }
    startTranspose = std::chrono::high_resolution_clock::now();
    trans(image);
    auto end = std::chrono::high_resolution_clock::now();
    transposeTime += std::chrono::duration_cast<std::chrono::milliseconds>(end - startTranspose).count();
    float totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Report results and statistics.
    imshow("Result", image);

    cout << "Time taken to get energy of each image: " << sobelEnergyTime << "(ms)" << endl;
    cout << "Time taken to get cumulative energy map: " << cumEnergyTime << "(ms)" << endl;
    cout << "Time taken to find seam: " << findSeamTime << "(ms)" << endl;
    cout << "Time taken to remove seam: " << removeSeamTime << "(ms)" << endl;
    cout << "Time taken to transpose image: " << transposeTime << "(ms)" << endl;
    cout << "Total time: " << totalTime << "(ms)" << endl;

    imwrite(outputName, image);
    waitKey(0);

    return 0;
}