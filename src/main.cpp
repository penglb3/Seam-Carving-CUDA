#include "getopt.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include<opencv2/opencv.hpp>
using namespace cv;
using std::string;

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

// https://github.com/ngwx1996/seam_carving

int main(int argc, char** argv)
{
    Mat img = imread("D:\\opencv\\sources\\samples\\data\\lena.jpg");

    std::cout 	<< "Image size= ("  << img.rows << "," 
                                    << img.cols << "," 
                                    << img.channels() 
                << "), type=" << type2str(img.type()) << std::endl;

    std::cout 	<< "BGR components in img[0,0]: " << (int) img.at<Vec3b>(0,0)[0] << ","
                                                << (int) img.at<Vec3b>(0,0)[1] << ","
                                                << (int) img.at<Vec3b>(0,0)[2] << std::endl;
    imshow("lena image",img);
    waitKey(0);
    destroyWindow("lena image");
    return 0;
}