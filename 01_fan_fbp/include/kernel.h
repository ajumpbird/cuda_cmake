#ifndef KERNEL
#define KERNEL

#include <cufft.h>
#include <cuda_runtime.h>
#include "tools.h"

__global__ void getInterpImg(float *d_interp_img, float theta, Geo geo, cudaTextureObject_t tex_obj);

__global__ void sumInterpImg(float *d_sino, float *d_interp_img, int view, Geo geo);

__global__ void padWeightSino(cufftReal *pad_weight_sino, float *sino, Geo geo);

__global__ void filterProj(cufftComplex *proj, cufftComplex *H, Geo geo);

__global__ void backProject(float *recon_img_d, Geo geo, float theta, cudaTextureObject_t tex_obj);

#endif // KERNEL