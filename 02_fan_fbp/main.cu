#define _USE_MATH_DEFINES
#include <iostream>
#include <cufft.h>
#include <cuda_runtime.h>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "tools.h"
#include "kernel.h"

using namespace std;
using namespace cv;

int main()
{
    // 设置几何参数
    Geo geo;
    geo.sod = 490.0f;
    geo.sdd = 880.0f;
    geo.detector_num = 2048;
    geo.detector_size = 0.2f;
    geo.views = 4000;
    geo.detector_length = geo.detector_num * geo.detector_size;
    geo.r = geo.sod * (geo.detector_length / 2) / sqrt((geo.detector_length / 2) * (geo.detector_length / 2) + geo.sdd * geo.sdd);
    geo.pixel_num = 2048;
    geo.pixel_size = 2 * geo.r / geo.pixel_num;
    geo.step_size = geo.pixel_size / 2;
    geo.padding_num = 2 * geo.detector_num;
    geo.da = 2 * M_PI / geo.views;

    // 记录开始时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 读取原始图像
    const char *image_path = "data/img.dat";
    int img_size = geo.pixel_num * geo.pixel_num;
    float *h_img = new float[img_size];
    readFile(image_path, h_img, img_size);
    // showImg(h_img, geo.pixel_num, geo.pixel_num, "Original Image", 512, 512);

    // 创建纹理对象，将图像数据绑定到纹理对象
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *cuArray;
    cudaMallocArray(&cuArray, &channelDesc, geo.pixel_num, geo.pixel_num);
    cudaTextureObject_t texObj = bindTexObj(cuArray, h_img, geo.pixel_num, geo.pixel_num);

    // sino 图
    int sino_size = geo.views * geo.detector_num;
    float *h_sino = new float[sino_size];
    float *d_sino;
    cudaMalloc(&d_sino, sino_size * sizeof(float));

    // 生成 sinogram
    dim3 block(16, 16);
    dim3 grid((geo.detector_num + block.x - 1) / block.x, (geo.views + block.y - 1) / block.y);
    getSino<<<grid, block>>>(d_sino, geo, texObj);
    cudaMemcpy(h_sino, d_sino, sino_size * sizeof(float), cudaMemcpyDeviceToHost);
    // showImg(h_sino, geo.views, geo.detector_num, "Sinogram", 512, 512);

    // 得到填充加权 sino
    int pws_size = geo.views * geo.padding_num;
    float *d_pws;
    cudaMalloc(&d_pws, pws_size * sizeof(float));
    dim3 block_pws(16, 16);
    dim3 grid_pws((geo.detector_num + block_pws.x - 1) / block_pws.x, geo.views, (geo.views + block_pws.y - 1) / block_pws.y);
    padWeightSino<<<grid_pws, block_pws>>>(d_pws, d_sino, geo);

    // 创建滤波器
    cufftComplex *H;
    cudaMalloc((void **)&H, (geo.padding_num / 2 + 1) * sizeof(cufftComplex));
    getH(H, geo);

    // rec_img 图
    float *h_rec_img = new float[img_size];
    memset(h_rec_img, 0, img_size * sizeof(float));
    float *d_rec_img;
    cudaMalloc(&d_rec_img, img_size * sizeof(float));
    cudaMemset(d_rec_img, 0, img_size * sizeof(float));

    // 重建图像
    recImg(d_pws, H, d_rec_img, geo);

    // 记录结束时间
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "Elapsed Time: " << elapsedTime << "ms" << endl;

    cudaMemcpy(h_rec_img, d_rec_img, img_size * sizeof(float), cudaMemcpyDeviceToHost);
    showImg(h_rec_img, geo.pixel_num, geo.pixel_num, "Reconstructed Image", 512, 512);

    // 释放内存
    delete[] h_img;
    delete[] h_sino;
    delete[] h_rec_img;
    
    cudaFree(d_sino);
    cudaFree(d_pws);
    cudaFree(H);
    cudaFree(d_rec_img);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}