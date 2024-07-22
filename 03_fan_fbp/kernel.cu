#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include "kernel.h"

__global__ void getSino(float *sino, Geo geo, cudaTextureObject_t tex_obj)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 第i个角度 行
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 第j个探测器 列
    if (i < geo.views && j < geo.detector_num)
    {
        float theta = i * geo.da; // 当前角度
        int iter_num = 2 * geo.r / geo.step_size + 1;
        float detector_y = geo.detector_length / 2 - geo.detector_size / 2 - j * geo.detector_size;
        float cos_phi = -geo.sdd / sqrt(detector_y * detector_y + geo.sdd * geo.sdd);
        float sin_phi = detector_y / sqrt(detector_y * detector_y + geo.sdd * geo.sdd);
        float sum = 0.0f;
        for (int k = 0; k < iter_num; k++)
        {
            float temp_length = geo.sod - geo.r + k * geo.step_size;
            float x = temp_length * cos_phi + geo.sod;
            float y = temp_length * sin_phi;
            float x_rot = x * cos(theta) - y * sin(theta);
            float y_rot = x * sin(theta) + y * cos(theta);
            float u = (x_rot - (-geo.r + geo.pixel_size / 2)) / geo.pixel_size + 0.5f;
            float v = (y_rot - (geo.r - geo.pixel_size / 2)) / (-geo.pixel_size) + 0.5f;
            sum += tex2D<float>(tex_obj, u, v);
        }
        sino[i * geo.detector_num + j] = sum * geo.step_size;
    }
}

__global__ void padWeightSino(cufftReal *pad_weight_sino, float *sino, Geo geo)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < geo.views && j < geo.detector_num)
    {
        float detector_y = geo.detector_length / 2 - geo.detector_size / 2 - j * geo.detector_size;
        pad_weight_sino[i * geo.padding_num + j] = sino[i * geo.detector_num + j] * geo.sdd / sqrt(geo.sdd * geo.sdd + detector_y * detector_y);
        pad_weight_sino[i * geo.padding_num + j + geo.detector_num] = 0.0f;
    }
}

__global__ void filterProj(cufftComplex *proj, cufftComplex *H, Geo geo)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 行
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 列
    if (i < geo.views && j < geo.padding_num / 2 + 1)
    {
        int idx = i * (geo.padding_num / 2 + 1) + j;
        float temp_x = proj[idx].x;
        float temp_y = proj[idx].y;
        proj[idx].x = temp_x * H[j].x - temp_y * H[j].y;
        proj[idx].y = temp_x * H[j].y + temp_y * H[j].x;
    }
}

__global__ void backProject(float *recon_img_d, Geo geo, float theta, cudaTextureObject_t tex_obj)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < geo.pixel_num && j < geo.pixel_num)
    {
        // 图像纹理坐标转换为空间坐标
        float x = -geo.r + geo.pixel_size / 2 + j * geo.pixel_size;
        float y = geo.r - geo.pixel_size / 2 - i * geo.pixel_size;
        float weight = geo.sod * geo.sdd / ((geo.sod - x * sin(theta) + y * cos(theta)) * (geo.sod - x * sin(theta) + y * cos(theta))) * geo.da / 2;
        // 旋转坐标
        float x_rot = x * cos(theta) + y * sin(theta);
        float y_rot = -x * sin(theta) + y * cos(theta);
        // 计算空间坐标对应的投影坐标
        float u = y_rot * geo.sdd / (geo.sod - x_rot);
        // 投影空间坐标转换为纹理坐标
        u = (u - (geo.detector_length / 2 - geo.detector_size / 2)) / (-geo.detector_size) + 0.5f;
        // 插值
        // delta_img[i * geo.pixel_num + j] = tex1D<float>(tex_obj, u) * weight;
        recon_img_d[i * geo.pixel_num + j] += tex1D<float>(tex_obj, u) * weight;
    }
}