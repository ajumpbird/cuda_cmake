#ifndef TOOLS
#define TOOLS

#include <cufft.h>
#include <cuda_runtime.h>

struct Geo
{
    float sod;
    float sdd;
    int detector_num;
    float detector_size;
    int views;
    float detector_length;
    float r;
    int pixel_num;
    float pixel_size;
    float step_size;
    int padding_num;
    float da;
};

void recImg(float *d_pws, cufftComplex *H, float *d_rec_img, Geo geo);

void getSino(float *d_sino, float *d_interp_img, Geo geo, cudaTextureObject_t texObj);

void getH(cufftComplex *H, Geo geo);

void readFile(const char *filename, float *data, int length);

void padData(cufftReal *input, int original_length, int padded_length);

void showImg(float *img, int rows, int cols, const char *winname, int resized_rows, int resized_cols);

cudaTextureObject_t bindTexObj(cudaArray *cuArray, float *array, int width, int height);

#endif // TOOLS