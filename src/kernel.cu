#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <float.h>
#include "parallel.h"

#define MAX_THREADS 1024
using namespace std;
using namespace cv;

extern float sobelEnergyTime;
extern float cumEnergyTime;
extern float findSeamTime;
extern float removeSeamTime;

/**************************************************************
* Error handling and timing
**************************************************************/
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
        exit( EXIT_FAILURE );
    }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define TIME_IT_CUDA( exec, time_recorder ) \
{\
    cudaEvent_t start, stop;\
    HANDLE_ERROR( cudaEventCreate(&start) );\
    HANDLE_ERROR( cudaEventCreate(&stop) );\
    cudaEventRecord(start);\
    exec;\
    cudaEventRecord(stop);\
    cudaEventSynchronize(stop);\
    HANDLE_ERROR( cudaEventElapsedTime(&time_recorder, start, stop) );\
}

/**************************************************************
* Kernel function headers.
**************************************************************/
__global__ void warm_up_gpu();
__global__ void energyKernel(const unsigned char* __restrict__ image, float* output, int rows, int cols);
__global__ void cudaEnergyMap(const unsigned char* __restrict__ energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize);
__global__ void cudaEnergyMapLarge(const unsigned char* __restrict__ energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current);
__global__ void cudaReduction(const unsigned char* __restrict__ row, float* mins, int* minsIndices, int size, int blockSize, int next);
__global__ void cudaRemoveSeam(unsigned char* image, int* seam, int rowSize, int colSize, int imageStep);

/**************************************************************
* Wrappers and utilities.
**************************************************************/
int nextPowerof2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}



void getEnergyMap(Mat& h_energy, Mat& h_energyMap, int rowSize, int colSize) {
    Mat h_prevEnergy;
    unsigned char* d_energy;
    unsigned char* d_energyMap;
    unsigned char* d_prevEnergy;
    int size = rowSize * colSize;

    HANDLE_ERROR( cudaMalloc(&d_energy, size * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc(&d_energyMap, size * sizeof(float)) );
    HANDLE_ERROR( cudaMemcpy(d_energy, h_energy.ptr(), size * sizeof(float), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(d_energyMap, h_energyMap.ptr(), size * sizeof(float), cudaMemcpyHostToDevice) );

    // Start from first row. Copy first row of energyMap to be used in device
    h_prevEnergy = Mat(1, colSize, CV_32F, float(0));
    h_energy.row(0).copyTo(h_prevEnergy.row(0));

    HANDLE_ERROR( cudaMalloc(&d_prevEnergy, colSize * sizeof(float)) );
    HANDLE_ERROR( cudaMemcpy(d_prevEnergy, h_prevEnergy.ptr(), colSize * sizeof(float), cudaMemcpyHostToDevice) );

    int blockSize = min(colSize, MAX_THREADS);
    int gridSize = ((colSize - 1) / MAX_THREADS) + 1;

    if (gridSize == 1) {
        cudaEnergyMap <<<gridSize, blockSize >>> (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize);
    }
    else {
        for (int i = 1; i < rowSize; i++) {
            cudaEnergyMapLarge <<<gridSize, blockSize >>> (d_energy, d_energyMap, d_prevEnergy, rowSize, colSize, i);
        }
    }


    HANDLE_ERROR( cudaMemcpy(h_energyMap.ptr(), d_energyMap, size * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaFree(d_energy) );
    HANDLE_ERROR( cudaFree(d_energyMap) );
    HANDLE_ERROR( cudaFree(d_prevEnergy) );
}


int getMinCumulativeEnergy(Mat& h_energyMap, int rowSize, int colSize) {
    // Require block size to be a multiple of 2 for parallel reduction
    // Sequential addressing ensures bank conflict free
    int blockSize;
    int gridSize;
    int lastSize;
    int sharedSize;
    Mat h_last;

    blockSize = min(nextPowerof2(colSize / 2), MAX_THREADS);
    gridSize = (nextPowerof2(colSize / 2) - 1) / MAX_THREADS + 1;
    sharedSize = blockSize * 2 * (sizeof(float) + sizeof(int));
    lastSize = colSize;
    // Copy last row of energyMap to be used in device
    h_last = Mat(1, colSize, CV_32F, float(0));
    h_energyMap.row(rowSize - 1).copyTo(h_last.row(0));

    // Allocate memory for host and device variables
    float* h_mins = new float[gridSize];
    int* h_minIndices = new int[gridSize];
    unsigned char* d_last;
    float* d_mins;
    int* d_minIndices;

    HANDLE_ERROR( cudaMalloc(&d_mins, gridSize * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc(&d_minIndices, gridSize * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc(&d_last, lastSize * sizeof(float)) );
    HANDLE_ERROR( cudaMemcpy(d_last, h_last.ptr(), lastSize * sizeof(float), cudaMemcpyHostToDevice) );

    cudaReduction << <gridSize, blockSize, sharedSize >> > (d_last, d_mins, d_minIndices, lastSize, blockSize, blockSize * gridSize);

    HANDLE_ERROR( cudaMemcpy(h_mins, d_mins, gridSize * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(h_minIndices, d_minIndices, gridSize * sizeof(int), cudaMemcpyDeviceToHost) );

    // Compare mins of different blocks
    pair<float, int> min = {h_mins[0], h_minIndices[0]};
    for (int i = 1; i < gridSize; i++) {
        if (min.first > h_mins[i]) {
            min.first = h_mins[i];
            min.second = h_minIndices[i];
        }
    }
    free(h_mins);
    free(h_minIndices);
    HANDLE_ERROR( cudaFree(d_last) );
    HANDLE_ERROR( cudaFree(d_mins) );
    HANDLE_ERROR( cudaFree(d_minIndices) );
    return min.second;
}

namespace CUDA{
    void warmUpGPU() {
        warm_up_gpu << <1, 1024 >> > ();
    }

    Mat createEnergyImg(Mat &image) {
        int rows = image.rows, cols = image.cols;
        Mat h_output(rows, cols, CV_32F, 0.), tmp;
        cvtColor(image, tmp, COLOR_BGR2GRAY);
        unsigned char* d_image;
        float* d_output;
        int size = tmp.rows * tmp.step;

        dim3 blockDim(32, 32);
        dim3 gridDim((tmp.cols + blockDim.x - 1) / blockDim.x, (tmp.rows + blockDim.y - 1) / blockDim.y);
        auto start = chrono::high_resolution_clock::now();

        HANDLE_ERROR( cudaMalloc(&d_image, size) );
        HANDLE_ERROR( cudaMalloc(&d_output, h_output.rows*h_output.step) );
        HANDLE_ERROR( cudaMemcpy(d_image, tmp.ptr(), size, cudaMemcpyHostToDevice) );

        //Kernel Call
        energyKernel <<< gridDim, blockDim >>> (d_image, d_output, rows, cols);
        
        HANDLE_ERROR( cudaMemcpy(h_output.ptr(), d_output, h_output.rows*h_output.step, cudaMemcpyDeviceToHost) );
        HANDLE_ERROR( cudaFree(d_image) );
        HANDLE_ERROR( cudaFree(d_output) );

        auto end = chrono::high_resolution_clock::now();
        sobelEnergyTime += chrono::duration_cast<chrono::milliseconds>(end - start).count();
        return h_output;
    }

    void removeSeam(Mat& h_image, vector<int> h_seam) {
        // dummy 1x1x3 to maintain matrix size;
        Mat h_output;
        unsigned char* d_image;
        int* d_seam;

        int rowSize = h_image.rows;
        int colSize = h_image.cols;
        int size = h_image.rows * h_image.step;
        dim3 blockDim(32, 32);
        dim3 gridDim((h_image.cols + blockDim.x - 1) / blockDim.x, (h_image.rows + blockDim.y - 1) / blockDim.y);
        auto startRemove = chrono::high_resolution_clock::now();

        HANDLE_ERROR( cudaMalloc(&d_image, size) );
        HANDLE_ERROR( cudaMalloc(&d_seam, h_seam.size() * sizeof(int)) );

        HANDLE_ERROR( cudaMemcpy(d_image, h_image.ptr(), size, cudaMemcpyHostToDevice) );
        HANDLE_ERROR( cudaMemcpy(d_seam, &h_seam[0], h_seam.size() * sizeof(int), cudaMemcpyHostToDevice) );

        cudaRemoveSeam << <gridDim, blockDim >> > (d_image, d_seam, rowSize, colSize, h_image.step);

        HANDLE_ERROR( cudaMemcpy(h_image.ptr(), d_image, size, cudaMemcpyDeviceToHost) );

        h_image = h_image.colRange(0, h_image.cols - 1);

        HANDLE_ERROR( cudaFree(d_image) );
        HANDLE_ERROR( cudaFree(d_seam) );

        auto endRemove = chrono::high_resolution_clock::now();
        removeSeamTime += chrono::duration_cast<chrono::milliseconds>(endRemove - startRemove).count();
    }
}

/**************************************************************
* Kernel functions
**************************************************************/
__global__ void warm_up_gpu() {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}

__global__ void energyKernel(const unsigned char* __restrict__ image, float* output, int rows, int cols){
    int tx = threadIdx.x + blockIdx.x * blockDim.x,
        ty = threadIdx.y + blockIdx.y * blockDim.y,
        here = ty*cols+tx;
    if (tx<cols && ty<rows){
        float   dy = image[(ty+1>=rows)?here:(cols+here)] - image[(ty==0)?here:(here-cols)],
                dx = image[(tx+1>=cols)?here:(here+1)] - image[(ty==0)?here:(here-1)];
        output[here] = (abs(dy)+abs(dx)) / 510.;
    }
}

__global__ void cudaEnergyMap(const unsigned char* __restrict__ energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize) {
    int idx;
    float topCenter, topLeft, topRight, minEnergy, cumEnergy;

    idx = blockIdx.x * MAX_THREADS + threadIdx.x;

    for (int current = 1; current < rowSize; current++) {
        if (idx < colSize) {
            // Find min value of prev row neighbors and add to the current idx's cumEnergy
            topCenter = ((float*)prevEnergy)[idx];
            topLeft = (idx > 0) ? ((float*)prevEnergy)[idx - 1] : ((float*)prevEnergy)[0];
            topRight = (idx < colSize - 1) ? ((float*)prevEnergy)[idx + 1] : ((float*)prevEnergy)[colSize - 1];
            minEnergy = min(topCenter, min(topLeft, topRight));
            cumEnergy = minEnergy + ((float*)energy)[current * colSize + idx];
        }
        __syncthreads();
        if (idx < colSize) {
            //Update cumEnergy in map and prevRow array
            ((float*)prevEnergy)[idx] = cumEnergy;
            ((float*)energyMap)[current * colSize + idx] = cumEnergy;
        }
        __syncthreads();
    }

}

__global__ void cudaEnergyMapLarge(const unsigned char* __restrict__ energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current) {
    int idx;
    float topCenter, topLeft, topRight, minEnergy, cumEnergy;

    idx = blockIdx.x * MAX_THREADS + threadIdx.x;

    if (idx >= colSize) {
        return;
    }
    // Find min value of prev row neighbors and add to the current idx's cumEnergy
    topCenter = ((float*)prevEnergy)[idx];
    topLeft = (idx > 0) ? ((float*)prevEnergy)[idx - 1] : ((float*)prevEnergy)[0];
    topRight = (idx < colSize - 1) ? ((float*)prevEnergy)[idx + 1] : ((float*)prevEnergy)[colSize - 1];
    minEnergy = min(topCenter, min(topLeft, topRight));
    cumEnergy = minEnergy + ((float*)energy)[current * colSize + idx];
    __syncthreads();
    //Update cumEnergy in map and prevRow array
    ((float*)prevEnergy)[idx] = cumEnergy;
    ((float*)energyMap)[current * colSize + idx] = cumEnergy;


}

__global__ void cudaReduction(const unsigned char* __restrict__ last, float* mins, int* minsIndices, int size, int blockSize, int next) {
    // Global index
    int idx = blockIdx.x * blockSize + threadIdx.x;
    // Initialize shared memory arrays
    extern __shared__ unsigned char sharedMemory[];
    float* sharedMins = (float*)sharedMemory;
    int* sharedMinIndices = (int*)(&(sharedMins[blockSize * 2]));
    
    // Since shared memory is shared in a block, the local idx is used while storing the value of the global idx cumEnergy
    sharedMins[threadIdx.x] = (idx < size) ? ((float*)last)[idx] : DBL_MAX;
    sharedMins[threadIdx.x + blockSize] = (idx + next < size) ? ((float*)last)[idx + next] : DBL_MAX;
    sharedMinIndices[threadIdx.x] = (idx < size) ? idx : INT_MAX;
    sharedMinIndices[threadIdx.x + blockSize] = (idx + next < size) ? idx + next : INT_MAX;

    __syncthreads();
    
    // Parallel reduction to get the min of the block
    for (int i = blockSize; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            if (sharedMins[threadIdx.x] > sharedMins[threadIdx.x + i]) {
                sharedMins[threadIdx.x] = sharedMins[threadIdx.x + i];
                sharedMinIndices[threadIdx.x] = sharedMinIndices[threadIdx.x + i];
            }
        }
        __syncthreads();
    }
    // local idx 0 has the min of the block
    if (threadIdx.x == 0) {
        mins[blockIdx.x] = sharedMins[0];
        minsIndices[blockIdx.x] = sharedMinIndices[0];
    }
}

__global__ void cudaRemoveSeam(unsigned char* image, int* seam, int rowSize, int colSize, int imageStep) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Location of colored pixel in input
    int tidImage = row * imageStep + (3 * col);
    float temp[3] = { 0 };

    if (col < colSize && row < rowSize) {
        if (col >= seam[row] && col != colSize - 1) {
            temp[0] = image[tidImage + 3];
            temp[1] = image[tidImage + 4];
            temp[2] = image[tidImage + 5];
        }
        else {
            temp[0] = image[tidImage];
            temp[1] = image[tidImage + 1];
            temp[2] = image[tidImage + 2];
        }
    }
    
    __syncthreads();
    if (col < colSize && row < rowSize) {
        image[tidImage] = temp[0];
        image[tidImage + 1] = temp[1];
        image[tidImage + 2] = temp[2];
    }
}
