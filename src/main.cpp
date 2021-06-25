#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>

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
float totalTime = 0;

int visualize = 0;
double fps = 30.0;
double spf = 1.0 / 30;
double* time_records;
int records_idx;
int cur_frame = 0;
VideoWriter out_capture;
int fW;
int fH;

int main(int argc, char** argv)
{    
    // Program parameters
    int width = 0, height = 0, parallel = 1;

    const char* imageName = "../images/Tension.jpg";
    const char* outputName = "../images/output.jpg";

    // Get parameters from arguments (if provided)
    char c;

    while((c = getopt(argc, argv, "w:h:i:o:sv:f:"))!=-1)
        switch(c){
            case 'w': width = atoi(optarg);break;		    // output width
            case 'h': height = atoi(optarg);break;		    // output height
            case 'i': imageName = optarg;break;			    // input filename
            case 'o': outputName = optarg;break;		    // output filename
            case 's': parallel = 0;break;				    // use parallel mode?
            case 'v': visualize = atoi(optarg);break;	    // visualize?
            case 'f': fps=atoi(optarg);spf=1.0/fps;break;   // fps
            default : abort();
        }
    if (!width && !height) {
        cout << "No resizing needed, exiting." << endl;
        return 0;
    }
    // Set up reduction width and height and read image.
    Mat image = imread(imageName, IMREAD_COLOR);
    if (image.empty()) {
        cout << "Invalid image. Please try again." << endl;
        return 1;
    }
    pair<int, int> imageSize = { image.cols, image.rows };

    int reduceWidth = (image.cols-width)%image.cols;
    int reduceHeight = (image.rows-height)%image.rows;

    printf(">>>>> Running %s <<<<<\n", parallel?"CUDA":"CPU");
    cout << "Image name: " << imageName << endl;
    cout << "Input dimension " << imageSize.first << " x " << imageSize.second << endl;

    // Choose the mode: default = CUDA
    void (*seamCarve)(Mat&, int&, int&) = CUDA::seamCarve;

    if (!parallel){
        seamCarve = CPU::seamCarve;
    }
    else
        CUDA::warmUpGPU();
    switch (visualize) {
        case 1:
            fH = fW = max(imageSize.first, imageSize.second);
            out_capture = VideoWriter("video.avi", VideoWriter::fourcc('M', 'J','P','G'), fps, Size(fW, fH));
            break;
        case 2:
            time_records = new double[reduceWidth + reduceHeight + 3];
            records_idx = -1;
    }
    seamCarve(image, reduceWidth, reduceHeight);
    if (visualize == 2) {
        visualize = 3;
        records_idx = -1;
        float temp_sobelEnergyTime = sobelEnergyTime;
        float temp_cumEnergyTime = cumEnergyTime;
        float temp_findSeamTime = findSeamTime;
        float temp_removeSeamTime = removeSeamTime;
        float temp_transposeTime = transposeTime;
        float temp_totalTime = totalTime;
        Mat temp_image = imread(imageName, IMREAD_COLOR);
        fH = fW = max(imageSize.first, imageSize.second);
        out_capture = VideoWriter("video.avi", VideoWriter::fourcc('M', 'J','P','G'), fps, Size(fW, fH));
        seamCarve = CUDA::seamCarve;
        seamCarve(temp_image, reduceWidth, reduceHeight);
        seamCarve = CPU::seamCarve;
        temp_image.release();
        sobelEnergyTime = temp_sobelEnergyTime;
        cumEnergyTime = temp_cumEnergyTime;
        findSeamTime = temp_findSeamTime;
        removeSeamTime = temp_removeSeamTime;
        transposeTime = temp_transposeTime;
        totalTime = temp_totalTime;
        delete time_records;
    }

    cout << "Output dimension " << image.cols << " x " << image.rows << endl << endl;

    // Report results and statistics.
    #ifdef _WIN32
    imshow("Result", image);
    #endif

    cout << "Time taken to get energy of each image: " << sobelEnergyTime << "(ms)" << endl;
    cout << "Time taken to get cumulative energy map: " << cumEnergyTime << "(ms)" << endl;
    cout << "Time taken to find seam: " << findSeamTime << "(ms)" << endl;
    cout << "Time taken to remove seam: " << removeSeamTime << "(ms)" << endl;
    cout << "Time taken to transpose image: " << transposeTime << "(ms)" << endl;
    cout << "Total time: " << totalTime << "(ms)" << endl;

    imwrite(outputName, image);

    return 0;
}
