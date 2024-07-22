#ifndef KERNEL
#define KERNEL

#include <cufft.h>
#include <cuda_runtime.h>
#include "tools.h"

__global__ void getSino(float *sino, Geo geo, cudaTextureObject_t tex_obj);

__global__ void padWeightSino(cufftReal *pad_weight_sino, float *sino, Geo geo);

__global__ void filterProj(cufftComplex *proj, cufftComplex *H, Geo geo);

__global__ void backProject(float *recon_img_d, Geo geo, float theta, cudaTextureObject_t tex_obj);

#endif // KERNEL