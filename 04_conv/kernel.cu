#include "kernel.h"

__global__ void compareImg(float *img1, float *img2, int row, int col, int *result)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row && j < col)
    {
        if (abs(img1[i * col + j] - img2[i * col + j]) > 1e-5)
        {
            *result = 0;
        }
    }
}

__global__ void convGPUSharedMem(float *img, float *mask, float *img_blur, int row, int col)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float img_sh[BLOCKDIM + MASK_WIDTH - 1][BLOCKDIM + MASK_WIDTH - 1];
    __shared__ float mask_sh[MASK_WIDTH][MASK_WIDTH];
    if (threadIdx.y < MASK_WIDTH && threadIdx.x < MASK_WIDTH)
    {
        mask_sh[threadIdx.y][threadIdx.x] = mask[threadIdx.y * MASK_WIDTH + threadIdx.x];
    }
    __syncthreads();
    int start_i = blockIdx.y * blockDim.y - MASK_WIDTH / 2;
    int start_j = blockIdx.x * blockDim.x - MASK_WIDTH / 2;
    int num = ((BLOCKDIM + MASK_WIDTH - 1) * (BLOCKDIM + MASK_WIDTH - 1) + BLOCKDIM * BLOCKDIM - 1) / (BLOCKDIM * BLOCKDIM);
    int thread_idx = threadIdx.y * BLOCKDIM + threadIdx.x;
    for (int k = 0; k < num; k++)
    {
        int img_sh_idx = k * BLOCKDIM * BLOCKDIM + thread_idx;
        if (img_sh_idx >= (BLOCKDIM + MASK_WIDTH - 1) * (BLOCKDIM + MASK_WIDTH - 1))
        {
            break;
        }
        int img_x = img_sh_idx / (BLOCKDIM + MASK_WIDTH - 1);
        int img_y = img_sh_idx % (BLOCKDIM + MASK_WIDTH - 1);
        int img_i = start_i + img_x;
        int img_j = start_j + img_y;
        if (img_i >= 0 && img_i < row && img_j >= 0 && img_j < col)
        {
            img_sh[img_x][img_y] = img[img_i * col + img_j];
        }
        else
        {
            img_sh[img_x][img_y] = 0;
        }
    }
    __syncthreads();
    float sum = 0;
    if (i < row && j < col)
    {
        for (int k = 0; k < MASK_WIDTH; k++)
        {
            for (int l = 0; l < MASK_WIDTH; l++)
            {
                sum += img_sh[threadIdx.y + k][threadIdx.x + l] * mask_sh[k][l];
            }
        }
        img_blur[i * col + j] = sum;
    }
}

__global__ void convGPUGlobalMem(float *img, float *mask, float *img_blur, int row, int col)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row && j < col)
    {
        float sum = 0;
        for (int k = 0; k < MASK_WIDTH; k++)
        {
            for (int l = 0; l < MASK_WIDTH; l++)
            {
                int img_row = i - MASK_WIDTH / 2 + k;
                int img_col = j - MASK_WIDTH / 2 + l;
                int img_idx = (img_row)*col + img_col;
                int mask_idx = k * MASK_WIDTH + l;
                if (img_row >= 0 && img_row < row && img_col >= 0 && img_col < col)
                {
                    sum += img[img_idx] * mask[mask_idx];
                }
            }
        }
        img_blur[i * col + j] = sum;
    }
}