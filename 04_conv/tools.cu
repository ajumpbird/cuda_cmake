#define _USE_MATH_DEFINES
#include <opencv2/opencv.hpp>
#include <math.h>
#include "tools.h"

using namespace cv;
using namespace std;

float recordGPUTime(void (*kernel)(float *, float *, float *, int, int), float *d_img, float *d_mask, float *d_img_blur_gpu, int img_rows, int img_cols, dim3 block, dim3 grid)
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

    // 记录起始事件
    err = cudaEventRecord(gpu_start, 0);
    if (err != cudaSuccess)
    {
        printf("Error recording start event: %s\n", cudaGetErrorString(err));
        return -1;
    }
    kernel<<<grid, block>>>(d_img, d_mask, d_img_blur_gpu, img_rows, img_cols);
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
    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);
    return gpu_elapsed;
}

void convCPU(float *img, float *mask, float *img_blur, int row, int col, int MASK_WIDTH)
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
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
}

void generateGaussFilter(float *mask, int mask_width)
{
    float sigma = 2.0f;
    float r, s = 2.0 * sigma * sigma;
    float sum = 0.0f;
    for (int i = 0; i < mask_width; i++)
    {
        for (int j = 0; j < mask_width; j++)
        {
            r = (i - mask_width / 2) * (i - mask_width / 2) + (j - mask_width / 2) * (j - mask_width / 2);
            mask[i * mask_width + j] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += mask[i * mask_width + j];
        }
    }
    for (int i = 0; i < mask_width; i++)
    {
        for (int j = 0; j < mask_width; j++)
        {
            mask[i * mask_width + j] /= sum;
        }
    }
}

void readFile(const char *filename, float *data, int length)
{
    FILE *file = fopen(filename, "rb");
    if (file == NULL)
    {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }
    fread(data, sizeof(float), length, file);
    fclose(file);
}

void showImg(float *img, int rows, int cols, const char *winname, int resized_rows, int resized_cols)
{
    Mat image(rows, cols, CV_32FC1, img);
    Mat normalized_image;
    normalize(image, normalized_image, 0, 1, NORM_MINMAX);
    Size target_size(resized_rows, resized_rows);
    Mat resized_image;
    resize(normalized_image, resized_image, target_size);
    imshow(winname, resized_image);
    waitKey(0);
}