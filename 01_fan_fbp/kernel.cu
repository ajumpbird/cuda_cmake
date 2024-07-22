#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include "kernel.h"

__global__ void getInterpImg(float *d_interp_img, float theta, Geo geo, cudaTextureObject_t tex_obj)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 行 第 i 个点
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 列 第 j 个探测器
    if (i < 2 * geo.r / geo.step_size + 1 && j < geo.detector_num)
    {
        float detector_y = geo.detector_length / 2 - geo.detector_size / 2 - j * geo.detector_size;
        float cos_phi = -geo.sdd / sqrt(geo.sdd * geo.sdd + detector_y * detector_y);
        float sin_phi = detector_y / sqrt(geo.sdd * geo.sdd + detector_y * detector_y);
        float temp_length = geo.sod - geo.r + i * geo.step_size;
        float x = temp_length * cos_phi + geo.sod;
        float y = temp_length * sin_phi;
        float x_rot = x * cos(theta) - y * sin(theta);
        float y_rot = x * sin(theta) + y * cos(theta);
        float u = (x_rot - (-geo.r + geo.pixel_size / 2)) / geo.pixel_size + 0.5f;
        float v = (y_rot - (geo.r - geo.pixel_size / 2)) / (-geo.pixel_size) + 0.5f;
        d_interp_img[i * geo.detector_num + j] = tex2D<float>(tex_obj, u, v);
    }
}

__global__ void sumInterpImg(float *d_sino, float *d_interp_img, int view, Geo geo)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 第 idx 个探测器
    if (idx < geo.detector_num)
    {
        float sum = 0.0f;
        for (int i = 0; i < 2 * geo.r / geo.step_size + 1; i++)
        {
            sum += d_interp_img[i * geo.detector_num + idx];
        }
        d_sino[view * geo.detector_num + idx] = sum * geo.step_size;
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < geo.padding_num / 2 + 1)
    {
        float temp_x = proj[idx].x;
        float temp_y = proj[idx].y;
        proj[idx].x = temp_x * H[idx].x - temp_y * H[idx].y;
        proj[idx].y = temp_x * H[idx].y + temp_y * H[idx].x;
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