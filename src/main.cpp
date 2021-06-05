#include "getopt.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include<opencv2/opencv.hpp>
using namespace cv;
int main(int argc, char** argv)
{
	Mat img = imread("D:\\opencv\\sources\\samples\\data\\lena.jpg");
	imshow("lena image",img);
	waitKey(0);
	destroyWindow("lena image");
	return 0;
}