#ifndef KERNEL
#define KERNEL

#include <cuda_runtime.h>

#define BLOCKDIM 16
#define MASK_WIDTH 19

__global__ void compareImg(float *img1, float *img2, int row, int col, int *result);

__global__ void convGPUSharedMem(float *img, float *mask, float *img_blur, int row, int col);

__global__ void convGPUGlobalMem(float *img, float *mask, float *img_blur, int row, int col);

#endif // KERNEL