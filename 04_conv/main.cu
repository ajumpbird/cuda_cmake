#define _USE_MATH_DEFINES
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "tools.h"
#include "kernel.h"

using namespace std;
using namespace cv;

int main()
{
    // 创建 CUDA 事件
    cudaError_t err;
    cudaEvent_t gpu_start, gpu_stop;
    err = cudaEventCreate(&gpu_start);
    if (err != cudaSuccess)
    {
        printf("Error creating start event: %s\n", cudaGetErrorString(err));
        return -1;
    }
    err = cudaEventCreate(&gpu_stop);
    if (err != cudaSuccess)
    {
        printf("Error creating stop event: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // 读取原始图像
    int img_rows = 2048;
    int img_cols = 2048;
    int img_size = img_rows * img_cols;
    float *h_img = new float[img_size];
    float *d_img;
    cudaMalloc(&d_img, img_size * sizeof(float));
    const char *img_filename = "data/img.dat";
    readFile(img_filename, h_img, img_size);
    cudaMemcpy(d_img, h_img, img_size * sizeof(float), cudaMemcpyHostToDevice);
    // showImg(h_img, img_rows, img_cols, "Original Image", 512, 512);

    // 生成高斯滤波器
    int mask_width = 19;
    float *h_mask = new float[mask_width * mask_width];
    float *d_mask;
    cudaMalloc(&d_mask, mask_width * mask_width * sizeof(float));
    generateGaussFilter(h_mask, mask_width);
    cudaMemcpy(d_mask, h_mask, mask_width * mask_width * sizeof(float), cudaMemcpyHostToDevice);

    // CPU 卷积
    // 获取起始时间
    auto cpu_start = std::chrono::high_resolution_clock::now();
    float *h_img_blur_cpu = new float[img_size];
    convCPU(h_img, h_mask, h_img_blur_cpu, img_rows, img_cols, mask_width);
    // 获取结束时间
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_elapsed = cpu_stop - cpu_start;
    cout << "CPU Time: " << cpu_elapsed.count() << "s" << endl;
    showImg(h_img_blur_cpu, img_rows, img_cols, "Blurred Image CPU", 512, 512);

    float *h_img_blur_gpu = new float[img_size];
    float *d_img_blur_gpu;
    cudaMalloc(&d_img_blur_gpu, img_size * sizeof(float));
    
    dim3 block(16, 16);
    dim3 grid((img_cols + block.x - 1) / block.x, (img_rows + block.y - 1) / block.y);
    // 记录起始事件
    err = cudaEventRecord(gpu_start, 0);
    if (err != cudaSuccess)
    {
        printf("Error recording start event: %s\n", cudaGetErrorString(err));
        return -1;
    }
    convGPUSharedMem<<<grid, block>>>(d_img, d_mask, d_img_blur_gpu, img_rows, img_cols);
    // convGPUGlobalMem<<<grid, block>>>(d_img, d_mask, d_img_blur_gpu, img_rows, img_cols);
    // 记录结束事件
    err = cudaEventRecord(gpu_stop, 0);
    if (err != cudaSuccess)
    {
        printf("Error recording stop event: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 等待事件完成
    err = cudaEventSynchronize(gpu_stop);
    if (err != cudaSuccess)
    {
        printf("Error synchronizing stop event: %s\n", cudaGetErrorString(err));
        return -1;
    }
    // 计算时间差
    float gpu_elapsed;
    err = cudaEventElapsedTime(&gpu_elapsed, gpu_start, gpu_stop);
    if (err != cudaSuccess)
    {
        printf("Error calculating elapsed time: %s\n", cudaGetErrorString(err));
        return -1;
    }
    cout << "GPU Time: " << gpu_elapsed << "ms" << endl;
    cudaMemcpy(h_img_blur_gpu, d_img_blur_gpu, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    showImg(h_img_blur_gpu, img_rows, img_cols, "Blurred Image GPU", 512, 512);
    
    // 比较图像
    for (int i = 0; i < img_size; i++)
    {
        if (abs(h_img_blur_cpu[i] - h_img_blur_gpu[i]) > 1e-3)
        {
            cout << "GPU Convolution is incorrect!" << endl;
            break;
        }
    }

    delete[] h_img;
    delete[] h_mask;
    delete[] h_img_blur_cpu;
    delete[] h_img_blur_gpu;

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);

    return 0;
}