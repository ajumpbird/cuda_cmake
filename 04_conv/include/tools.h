#ifndef TOOLS
#define TOOLS

#include <cuda_runtime.h>

// 时间会比不封装更长
float recordGPUTime(void (*kernel)(float *, float *, float *, int, int), float *img, float *mask, float *img_blur, int row, int col, dim3 block, dim3 grid);

/**
 * @brief CPU版本的卷积操作，通过if判断边界条件
 * @param img 输入图像
 * @param mask 卷积核
 * @param img_blur 输出图像
 * @param row 图像行数
 * @param col 图像列数
 * @param MASK_WIDTH 卷积核宽度
 */
void convCPU(float *img, float *mask, float *img_blur, int row, int col, int MASK_WIDTH);

void generateGaussFilter(float *mask, int mask_width);

void readFile(const char *filename, float *data, int length);

void showImg(float *img, int rows, int cols, const char *winname, int resized_rows, int resized_cols);

#endif // TOOLS