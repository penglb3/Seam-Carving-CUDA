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
* Error handling
**************************************************************/
static void HandleError( cudaError_t err, const char *file, int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString( err ),
        file, line );
        exit( EXIT_FAILURE );
    }
}


#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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
__global__ void warmUpKernel();
__global__ void transposeKernel(const uchar3* __restrict__ image, uchar3* out, int rows, int cols);
__global__ void energyKernel(const uchar3* __restrict__ image, float* output, int rows, int cols);
__global__ void energyMapKernel_1B(const float* __restrict__ energy, float* energyMap, float* prevEnergy, int rows, int cols);
__global__ void energyMapKernel_nB(const float* __restrict__ energy, float* energyMap, float* prevEnergy, int rows, int cols, int current);
__global__ void minReductionKernel(const float* __restrict__ row, float* mins, int* minsIndices, int size);
__global__ void removeSeamKernel(const uchar3* __restrict__ image, uchar3* out, int* seam, int rows, int cols);

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

namespace CUDA{
    void warmUpGPU() {
        warmUpKernel << <1, 1024 >> > ();
    }

    void trans(GpuMat& d_image){
        int rows = d_image.rows;
        int cols = d_image.cols;

        dim3 blockDim(32, 32);
        dim3 gridDim((d_image.cols + blockDim.x - 1) / blockDim.x, (d_image.rows + blockDim.y - 1) / blockDim.y);

        GpuMat d_out(cols, rows, d_image.type());

        transposeKernel<<<gridDim, blockDim>>>(d_image.ptr<uchar3>(), d_out.ptr<uchar3>(), rows, cols);

        d_image.release();
        d_image = d_out;
    }

    GpuMat calculateEnergyImg(GpuMat &d_image) {
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

        int rows = d_image.rows;
        int cols = d_image.cols;

        GpuMat d_out(rows, cols-1, CV_8UC3);

        dim3 blockDim(32, 32);
        dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y);
        auto startRemove = chrono::high_resolution_clock::now();

        HANDLE_ERROR( cudaMalloc(&d_seam, h_seam.size() * sizeof(int)) );
        HANDLE_ERROR( cudaMemcpy(d_seam, h_seam.data(), h_seam.size() * sizeof(int), cudaMemcpyHostToDevice) );

        removeSeamKernel << <gridDim, blockDim >> > (d_image.ptr<uchar3>(), d_out.ptr<uchar3>(), d_seam, rows, cols);

        d_image = d_out;

        HANDLE_ERROR( cudaFree(d_seam) );

        auto endRemove = chrono::high_resolution_clock::now();
        removeSeamTime += chrono::duration_cast<chrono::microseconds>(endRemove - startRemove).count() / 1e3;
    }
}


void getEnergyMap(GpuMat& d_energy, GpuMat& d_energyMap, int rows, int cols) {

    // Start from first row. Copy first row of energyMap to be used in device
    GpuMat d_prevEnergy(1, cols, CV_32F, float(0));
    d_energy.row(0).copyTo(d_prevEnergy.row(0));

    int blockSize = min(cols, MAX_THREADS);
    int gridSize = ((cols - 1) / MAX_THREADS) + 1;

    if (gridSize == 1) {
        energyMapKernel_1B <<<gridSize, blockSize >>> (d_energy.ptr<float>(), d_energyMap.ptr<float>(), d_prevEnergy.ptr<float>(), rows, cols);
    }
    else {
        for (int i = 1; i < rows; i++) {
            energyMapKernel_nB <<<gridSize, blockSize >>> (d_energy.ptr<float>(), d_energyMap.ptr<float>(), d_prevEnergy.ptr<float>(), rows, cols, i);
        }
    }

    d_prevEnergy.release();
}


int getMinCumulativeEnergy(GpuMat& d_energyMap) {
    // Require block size to be a multiple of 2 for parallel reduction
    // Sequential addressing ensures bank conflict free
    int rows = d_energyMap.rows, cols = d_energyMap.cols;
    int blockSize = min(nextPowerof2(cols / 2), MAX_THREADS);
    int gridSize = (nextPowerof2(cols / 2) - 1) / MAX_THREADS + 1;
    int sharedMemSize = blockSize * 2 * (sizeof(float) + sizeof(int));

    // Copy last row of energyMap to be used in device
    GpuMat d_last(1, cols, CV_32F, float(0));
    d_energyMap.row(rows - 1).copyTo(d_last.row(0));

    // Allocate memory for host and device variables
    float* h_minValues = new float[gridSize];
    int* h_minIndices = new int[gridSize];
    float* d_minValues;
    int* d_minIndices;

    HANDLE_ERROR( cudaMalloc(&d_minValues, gridSize * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc(&d_minIndices, gridSize * sizeof(int)) );

    minReductionKernel <<<gridSize, blockSize, sharedMemSize >>>(d_last.ptr<float>(), d_minValues, d_minIndices, cols);

    HANDLE_ERROR( cudaMemcpy(h_minValues, d_minValues, gridSize * sizeof(float), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy(h_minIndices, d_minIndices, gridSize * sizeof(int), cudaMemcpyDeviceToHost) );

    // Compare mins of different blocks
    pair<float, int> min = {h_minValues[0], h_minIndices[0]};
    for (int i = 1; i < gridSize; i++) {
        if (min.first > h_minValues[i]) {
            min.first = h_minValues[i];
            min.second = h_minIndices[i];
        }
    }
    free(h_minValues);
    free(h_minIndices);
    
    d_last.release();
    HANDLE_ERROR( cudaFree(d_minValues) );
    HANDLE_ERROR( cudaFree(d_minIndices) );
    return min.second;
}


/**************************************************************
* Kernel functions
**************************************************************/
__global__ void warmUpKernel() {
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
        uchar3  left = image[(ty==0)?here:(here-1)],
                right = image[(tx+1>=cols)?here:(here+1)],
                down = image[(ty+1>=rows)?here:(cols+here)],
                up = image[(ty==0)?here:(here-cols)];
        float   dy = bgr2gray(down) - bgr2gray(up),
                dx = bgr2gray(right) - bgr2gray(left);
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


__global__ void energyMapKernel_1B(const float* __restrict__ energy, float* energyMap, float* prevEnergy, int rows, int cols) {
    int idx;
    float upper, upperLeft, upperRight, minEnergy, cumEnergy;

    idx = blockIdx.x * MAX_THREADS + threadIdx.x;

    for (int current = 1; current < rows; current++) {
        if (idx < cols) {
            // Find min value of prev row neighbors and add to the current idx's cumEnergy
            upper = prevEnergy[idx];
            upperLeft = (idx > 0) ? prevEnergy[idx - 1] : prevEnergy[0];
            upperRight = (idx < cols - 1) ? prevEnergy[idx + 1] : prevEnergy[cols - 1];
            minEnergy = min(upper, min(upperLeft, upperRight));
            cumEnergy = minEnergy + energy[current * cols + idx];
        }
        __syncthreads();
        if (idx < cols) {
            //Update cumEnergy in map and prevRow array
            prevEnergy[idx] = cumEnergy;
            energyMap[current * cols + idx] = cumEnergy;
        }
        __syncthreads();
    }

}

__global__ void energyMapKernel_nB(const float* __restrict__ energy, float* energyMap, float* prevEnergy, int rows, int cols, int current) {
    int idx;
    float upper, upperLeft, upperRight, minEnergy, cumEnergy;

    idx = blockIdx.x * MAX_THREADS + threadIdx.x;

    if (idx >= cols) {
        return;
    }
    // Find min value of prev row neighbors and add to the current idx's cumEnergy
    upper = prevEnergy[idx];
    upperLeft = (idx > 0) ? prevEnergy[idx - 1] : prevEnergy[0];
    upperRight = (idx < cols - 1) ? prevEnergy[idx + 1] : prevEnergy[cols - 1];
    minEnergy = min(upper, min(upperLeft, upperRight));
    cumEnergy = minEnergy + energy[current * cols + idx];
    __syncthreads();
    //Update cumEnergy in map and prevRow array
    prevEnergy[idx] = cumEnergy;
    energyMap[current * cols + idx] = cumEnergy;


}

__global__ void minReductionKernel(const float* __restrict__ row, float* mins, int* minsIndices, int size) {
    // Global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int next = blockDim.x * gridDim.x;
    // Initialize shared memory arrays
    extern __shared__ unsigned char sharedMemory[];
    float* s_minValues = (float*)sharedMemory;
    int* s_minIndices = (int*)(s_minValues + blockDim.x * 2);
    
    // Since shared memory is shared in a block, the local idx is used while storing the value of the global idx cumEnergy
    s_minValues[threadIdx.x] = (idx < size) ? row[idx] : DBL_MAX;
    s_minValues[threadIdx.x + blockDim.x] = (idx + next < size) ? row[idx + next] : DBL_MAX;
    s_minIndices[threadIdx.x] = (idx < size) ? idx : INT_MAX;
    s_minIndices[threadIdx.x + blockDim.x] = (idx + next < size) ? idx + next : INT_MAX;

    __syncthreads();
    
    // Parallel reduction to get the min of the block
    for (int s = blockDim.x; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (s_minValues[threadIdx.x] > s_minValues[threadIdx.x + s]) {
                s_minValues[threadIdx.x] = s_minValues[threadIdx.x + s];
                s_minIndices[threadIdx.x] = s_minIndices[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    // local idx 0 has the min of the block
    if (threadIdx.x == 0) {
        mins[blockIdx.x] = s_minValues[0];
        minsIndices[blockIdx.x] = s_minIndices[0];
    }
}

__global__ void removeSeamKernel (const uchar3* __restrict__ image, uchar3* out, int* seam, int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Location of colored pixel in input
    int from = row * cols + col,
        to = row * (cols - 1) + col;
    uchar3 pixel;

    if (col < cols && row < rows) {
        pixel = (col>=seam[row] && col!=cols-1)? image[from+1]: image[from];
    }
    
    __syncthreads();
    if (col < cols - 1 && row < rows) {
        out[to] = pixel;
    }
}
