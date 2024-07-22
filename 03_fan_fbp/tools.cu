#define _USE_MATH_DEFINES
#include <math.h>
#include <opencv2/opencv.hpp>
#include "tools.h"
#include "kernel.h"

using namespace cv;

void recImg(float *d_pws, cufftComplex *H, float *d_rec_img, Geo geo)
{
    // 批量投影数据
    cufftComplex *d_proj_batch;
    cudaMalloc((void **)&d_proj_batch, geo.views * (geo.padding_num / 2 + 1) * sizeof(cufftComplex));
    cufftReal *d_real_proj_batch;
    cudaMalloc((void **)&d_real_proj_batch, geo.views * geo.padding_num * sizeof(cufftReal));
    cufftReal *h_real_proj_batch = new cufftReal[geo.views * geo.padding_num];
    cufftReal *h_truncated_proj_batch = new cufftReal[geo.views * geo.detector_num];

    // 创建纹理对象，将投影数据绑定到纹理对象
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *cuArray_proj;
    cudaMallocArray(&cuArray_proj, &channelDesc, geo.detector_num, 1);
    cudaTextureObject_t texObj_proj = bindTexObj(cuArray_proj, h_truncated_proj_batch, geo.detector_num, 1);

    // 得到加权填充投影的傅里叶变换
    cufftHandle plan_r2c;
    cufftPlan1d(&plan_r2c, geo.padding_num, CUFFT_R2C, geo.views);
    cufftExecR2C(plan_r2c, d_pws, d_proj_batch);
    cufftDestroy(plan_r2c);
    // 滤波
    dim3 block_filterProj(16, 16);
    dim3 grid_filterProj((geo.padding_num / 2 + 1 + block_filterProj.x - 1) / block_filterProj.x, (geo.views + block_filterProj.y - 1) / block_filterProj.y);
    filterProj<<<grid_filterProj, block_filterProj>>>(d_proj_batch, H, geo);
    // 傅里叶逆变换
    cufftHandle plan_c2r;
    cufftPlan1d(&plan_c2r, geo.padding_num, CUFFT_C2R, geo.views);
    cufftExecC2R(plan_c2r, d_proj_batch, d_real_proj_batch);
    cufftDestroy(plan_c2r);
    cudaMemcpy(h_real_proj_batch, d_real_proj_batch, geo.views * geo.padding_num * sizeof(cufftReal), cudaMemcpyDeviceToHost);
    // 归一化
    for (int j = 0; j < geo.views; j++)
    {
        for (int k = 0; k < geo.padding_num; k++)
        {
            h_real_proj_batch[j * geo.padding_num + k] /= geo.padding_num;
        }
    }
    // 截取
    for (int j = 0; j < geo.views; j++)
    {
        for (int k = 0; k < geo.detector_num; k++)
        {
            h_truncated_proj_batch[j * geo.detector_num + k] = h_real_proj_batch[j * geo.padding_num + k + geo.detector_num / 2];
        }
    }

    dim3 block_rec(16, 16);
    dim3 grid_rec((geo.pixel_num + block_rec.x - 1) / block_rec.x, (geo.pixel_num + block_rec.y - 1) / block_rec.y);
    for (int i = 0; i < geo.views; i++)
    {
        // 投影数据绑定到纹理对象
        cudaMemcpyToArray(cuArray_proj, 0, 0, &h_truncated_proj_batch[i * geo.detector_num], geo.detector_num * sizeof(cufftReal), cudaMemcpyHostToDevice);
        // 重建
        backProject<<<grid_rec, block_rec>>>(d_rec_img, geo, i * geo.da, texObj_proj);
    }

    delete[] h_real_proj_batch;
    delete[] h_truncated_proj_batch;
    cudaFree(d_proj_batch);
    cudaFree(d_real_proj_batch);
}

void getH(cufftComplex *H, Geo geo)
{
    cufftReal *h_filter = new cufftReal[geo.padding_num];
    cufftReal *d_filter;
    cudaMalloc((void **)&d_filter, geo.padding_num * sizeof(cufftReal));
    for (int i = 0; i < geo.detector_num; i++)
    {
        h_filter[i] = -1 / ((M_PI * (i - geo.detector_num / 2 - 1) * geo.detector_size) * (M_PI * (i - geo.detector_num / 2 - 1) * geo.detector_size));
        if ((i - geo.detector_num / 2 - 1) % 2 == 0)
        {
            h_filter[i] = 0.0f;
        }
    }
    h_filter[geo.detector_num / 2 + 1] = 1 / (4 * geo.detector_size * geo.detector_size);
    padData(h_filter, geo.detector_num, geo.padding_num);
    cudaMemcpy(d_filter, h_filter, geo.padding_num * sizeof(cufftReal), cudaMemcpyHostToDevice);

    cufftHandle plan_r2c;
    cufftPlan1d(&plan_r2c, geo.padding_num, CUFFT_R2C, 1);
    cufftExecR2C(plan_r2c, d_filter, H);

    cufftDestroy(plan_r2c);
    delete[] h_filter;
    cudaFree(d_filter);
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

// 对实信号进行填充0
void padData(cufftReal *input, int original_length, int padded_length)
{
    for (int i = original_length; i < padded_length; i++)
    {
        input[i] = 0.0f;
    }
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

cudaTextureObject_t bindTexObj(cudaArray *cuArray, float *array, int width, int height)
{
    // 创建纹理对象，将图像数据绑定到纹理对象
    // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    // cudaArray *cuArray;
    // cudaMallocArray(&cuArray, &channelDesc, width, height);
    cudaMemcpyToArray(cuArray, 0, 0, array, width * height * sizeof(float), cudaMemcpyHostToDevice); // 将原始图像的数据拷贝到cuArray
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray; // 将cuArray赋值给resDesc
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}