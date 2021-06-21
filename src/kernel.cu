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
using namespace cv::cuda;

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
* Definitions of the allocator without alignment.
**************************************************************/
bool MyAllocator::allocate(GpuMat* mat, int rows, int cols, size_t elemSize)
{
    // Single row or single column must be continuous
    HANDLE_ERROR( cudaMalloc(&mat->data, elemSize * cols * rows) );
    mat->step = elemSize * cols;

    mat->refcount = (int*) fastMalloc(sizeof(int));

    return true;
}

void MyAllocator::free(GpuMat* mat)
{
    cudaFree(mat->datastart);
    fastFree(mat->refcount);
}

/**************************************************************
* Kernel function headers.
**************************************************************/
__global__ void warm_up_gpu();
__global__ void transposeKernel(const uchar3* __restrict__ image, uchar3* out, int rows, int cols);
__global__ void energyKernel(const uchar3* __restrict__ image, float* output, int rows, int cols);
__global__ void cudaEnergyMap(const unsigned char* __restrict__ energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize);
__global__ void cudaEnergyMapLarge(const unsigned char* __restrict__ energy, unsigned char* energyMap, unsigned char* prevEnergy, int rowSize, int colSize, int current);
__global__ void cudaReduction(const unsigned char* __restrict__ row, float* mins, int* minsIndices, int size, int blockSize, int next);
__global__ void cudaRemoveSeam(uchar3* image, int* seam, int rowSize, int colSize);

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


void getEnergyMap(GpuMat& d_energy, GpuMat& d_energyMap, int rowSize, int colSize) {

    // Start from first row. Copy first row of energyMap to be used in device
    GpuMat d_prevEnergy(1, colSize, CV_32F, float(0));
    d_energy.row(0).copyTo(d_prevEnergy.row(0));

    int blockSize = min(colSize, MAX_THREADS);
    int gridSize = ((colSize - 1) / MAX_THREADS) + 1;

    if (gridSize == 1) {
        cudaEnergyMap <<<gridSize, blockSize >>> (d_energy.ptr<unsigned char>(), d_energyMap.ptr<unsigned char>(), d_prevEnergy.ptr<unsigned char>(), rowSize, colSize);
    }
    else {
        for (int i = 1; i < rowSize; i++) {
            cudaEnergyMapLarge <<<gridSize, blockSize >>> (d_energy.ptr<unsigned char>(), d_energyMap.ptr<unsigned char>(), d_prevEnergy.ptr<unsigned char>(), rowSize, colSize, i);
        }
    }

    d_prevEnergy.release();
}


int getMinCumulativeEnergy(GpuMat& d_energyMap, int rowSize, int colSize) {
    // Require block size to be a multiple of 2 for parallel reduction
    // Sequential addressing ensures bank conflict free
    int blockSize;
    int gridSize;
    int lastSize;
    int sharedSize;

    blockSize = min(nextPowerof2(colSize / 2), MAX_THREADS);
    gridSize = (nextPowerof2(colSize / 2) - 1) / MAX_THREADS + 1;
    sharedSize = blockSize * 2 * (sizeof(float) + sizeof(int));
    lastSize = colSize;
    // Copy last row of energyMap to be used in device
    GpuMat d_last(1, colSize, CV_32F, float(0));
    d_energyMap.row(rowSize - 1).copyTo(d_last.row(0));

    // Allocate memory for host and device variables
    float* h_mins = new float[gridSize];
    int* h_minIndices = new int[gridSize];
    float* d_mins;
    int* d_minIndices;

    HANDLE_ERROR( cudaMalloc(&d_mins, gridSize * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc(&d_minIndices, gridSize * sizeof(int)) );

    cudaReduction <<<gridSize, blockSize, sharedSize >>>(d_last.ptr<unsigned char>(), d_mins, d_minIndices, lastSize, blockSize, blockSize * gridSize);

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
    
    d_last.release();
    HANDLE_ERROR( cudaFree(d_mins) );
    HANDLE_ERROR( cudaFree(d_minIndices) );
    return min.second;
}

namespace CUDA{
    void warmUpGPU() {
        warm_up_gpu << <1, 1024 >> > ();
    }

    void trans(GpuMat& d_image){
        int rowSize = d_image.rows;
        int colSize = d_image.cols;

        dim3 blockDim(32, 32);
        dim3 gridDim((d_image.cols + blockDim.x - 1) / blockDim.x, (d_image.rows + blockDim.y - 1) / blockDim.y);

        GpuMat d_out(colSize, rowSize, d_image.type());

        transposeKernel<<<gridDim, blockDim>>>(d_image.ptr<uchar3>(), d_out.ptr<uchar3>(), rowSize, colSize);

        d_image.release();
        d_image = d_out;
    }

    GpuMat createEnergyImg(GpuMat &d_image) {
        auto start = chrono::high_resolution_clock::now();
        int rows = d_image.rows, cols = d_image.cols;

        GpuMat d_output(rows, cols, CV_32F, 0.);

        dim3 blockDim(32, 32);
        dim3 gridDim((d_image.cols + blockDim.x - 1) / blockDim.x, (d_image.rows + blockDim.y - 1) / blockDim.y);

        energyKernel <<< gridDim, blockDim >>> (d_image.ptr<uchar3>(), d_output.ptr<float>(), rows, cols);

        auto end = chrono::high_resolution_clock::now();
        sobelEnergyTime += chrono::duration_cast<chrono::microseconds>(end - start).count() / 1e3;

        return d_output;
    }

    void removeSeam(GpuMat& d_image, vector<int> h_seam) {
        int* d_seam;

        int rowSize = d_image.rows;
        int colSize = d_image.cols;

        dim3 blockDim(32, 32);
        dim3 gridDim((colSize + blockDim.x - 1) / blockDim.x, (rowSize + blockDim.y - 1) / blockDim.y);
        auto startRemove = chrono::high_resolution_clock::now();

        HANDLE_ERROR( cudaMalloc(&d_seam, h_seam.size() * sizeof(int)) );
        HANDLE_ERROR( cudaMemcpy(d_seam, &h_seam[0], h_seam.size() * sizeof(int), cudaMemcpyHostToDevice) );

        cudaRemoveSeam << <gridDim, blockDim >> > (d_image.ptr<uchar3>(), d_seam, rowSize, colSize);

        d_image = d_image.colRange(0, d_image.cols - 1).clone();

        HANDLE_ERROR( cudaFree(d_seam) );

        auto endRemove = chrono::high_resolution_clock::now();
        removeSeamTime += chrono::duration_cast<chrono::microseconds>(endRemove - startRemove).count() / 1e3;
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


inline __device__ __host__ char bgr2gray(const uchar3 bgr){
    return 0.299f * bgr.z + 0.587f * bgr.y + 0.114f * bgr.x;
}

__global__ void energyKernel(const uchar3* __restrict__ image, float* output, int rows, int cols){
    int tx = threadIdx.x + blockIdx.x * blockDim.x,
        ty = threadIdx.y + blockIdx.y * blockDim.y,
        here = ty*cols+tx;
    if (tx<cols && ty<rows){
        float   dy = bgr2gray(image[(ty+1>=rows)?here:(cols+here)]) - bgr2gray(image[(ty==0)?here:(here-cols)]),
                dx = bgr2gray(image[(tx+1>=cols)?here:(here+1)]) - bgr2gray(image[(ty==0)?here:(here-1)]);
        output[here] = (abs(dy)+abs(dx)) / 510.;
    }
}

__global__ void transposeKernel(const uchar3* __restrict__ image, uchar3* out, int rows, int cols){
    __shared__ uchar3 smem[32][32];

    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x,
        iy = blockIdx.y * blockDim.y + threadIdx.y;
    if ((ix<cols) && (iy<rows)){
        smem[threadIdx.y][threadIdx.x] = image[iy*cols+ix];
    }
    __syncthreads();
    ix = blockIdx.y * blockDim.y + threadIdx.x,
    iy = blockIdx.x * blockDim.x + threadIdx.y;
    if ((ix<rows) && (iy<cols)){
        out[iy*rows+ix] = smem[threadIdx.x][threadIdx.y];
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

__global__ void cudaRemoveSeam (uchar3 * image, int* seam, int rowSize, int colSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Location of colored pixel in input
    int tidImage = row * colSize + col;
    uchar3 temp;

    if (col < colSize && row < rowSize) {
        if (col >= seam[row] && col != colSize - 1) {
            temp = image[tidImage+1];
        }
        else {
            temp = image[tidImage];
        }
    }
    
    __syncthreads();
    if (col < colSize && row < rowSize) {
        image[tidImage] = temp;
    }
}
