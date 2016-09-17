#include "../../../headers/activationFunction.h"

/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */



/*
 * This sample demonstrates two adaptive image denoising technqiues:
 * KNN and NLM, based on computation of both geometric and color distance
 * between texels. While both techniques are already implemented in the
 * DirectX SDK using shaders, massively speeded up variation
 * of the latter techique, taking advantage of shared memory, is implemented
 * in addition to DirectX counterparts.
 * See supplied whitepaper for more explanations.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "imageBinarization.h"


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
#define THRESHOLD 127
__device__ unsigned char __max(unsigned char x, unsigned char y)
{
    return (x > y) ? x : y;
}

__device__  unsigned char __min(unsigned char x, unsigned char y)
{
    return (x < y) ? x : y;
}

int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// __device__ float lerpf(float a, float b, float c)
// {
//     return a + (b - a) * c;
// }

// __device__ float vecLen(float4 a, float4 b)
// {
//     return (
//                (b.x - a.x) * (b.x - a.x) +
//                (b.y - a.y) * (b.y - a.y) +
//                (b.z - a.z) * (b.z - a.z)
//            );
// }

__device__ TColor make_color(unsigned char r, unsigned char g, unsigned char b, unsigned char a)
{
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}



////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
//texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
texture<uchar4, 2, cudaReadModeElementType> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
cudaArray *a_Src;

////////////////////////////////////////////////////////////////////////////////
// Filtering kernels
////////////////////////////////////////////////////////////////////////////////
// #include "imageDenoising_copy_kernel.cuh"
// #include "imageDenoising_nlm_kernel.cuh"
// #include "imageDenoising_nlm2_kernel.cuh"




//-----------------------------------------------------------------------------
/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
//The macro CUPRINTF is defined for architectures
//with different compute capabilities.
#if __CUDA_ARCH__ < 200     //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                       //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)
#endif


////////////////////////////////////////////////////////////////////////////////
// image binarization kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void imageBinarization(
    unsigned char *dst,
    int imageW,
    int imageH
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < imageW && iy < imageH)
    {

        float parrotInput[3];
        float parrotOutput[1];

        uchar4 currPixel = tex2D(texImage, ix, iy);
        unsigned char b  = (unsigned char) currPixel.z;
        unsigned char g  = (unsigned char) currPixel.y;
        unsigned char r  = (unsigned char) currPixel.x;


        parrotInput[0] = r / 255.0;
        parrotInput[1] = g / 255.0;
        parrotInput[2] = b / 255.0;

        unsigned char result;
float layer_1_0 = parrotInput[0] * 4.699794 + parrotInput[1] * 0.877188 + parrotInput[2] * 0.496408 + 1.0f * -3.661214;

float layer_1_1 = parrotInput[0] * -3.402053 + parrotInput[1] * 3.861662 + parrotInput[2] * -6.814451 + 1.0f * 1.839221;

float layer_1_2 = parrotInput[0] * -2.248324 + parrotInput[1] * -8.565042 + parrotInput[2] * 3.407546 + 1.0f * 1.899614;

float layer_1_3 = parrotInput[0] * 4.647291 + parrotInput[1] * 0.815243 + parrotInput[2] * 0.595827 + 1.0f * -3.638438;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * -10.469186 + sigmoid(layer_1_1, 0.500000) * 24.400442 + sigmoid(layer_1_2, 0.500000) * 24.699705 + sigmoid(layer_1_3, 0.500000) * -10.481432 + 1.0f * 0.666498;

layer_2_0 = sigmoid(layer_2_0, 0.5);

float layer_2_1 = sigmoid(layer_1_0, 0.500000) * -9.471014 + sigmoid(layer_1_1, 0.500000) * 19.837952 + sigmoid(layer_1_2, 0.500000) * 20.186312 + sigmoid(layer_1_3, 0.500000) * -9.508874 + 1.0f * 1.531181;

layer_2_1 = sigmoid(layer_2_1, 0.5);

float layer_3_0 = sigmoid(layer_2_0, 0.500000) * -10.481432 + sigmoid(layer_2_1, 0.000000) * 0.666498 + 1.0f * -3.288635;

layer_3_0 = sigmoid(layer_3_0, 0.5);

parrotOutput[0] = layer_3_0;

// parrotOutput[0] = layer_3_0;
// 
//         unsigned char mi = __min(r, __min(g, b));
//         unsigned char ma = __max(r, __max(g, b));
// 
//         result = (((unsigned short) ma + (unsigned short) mi) > THRESHOLD * 2 ) ? 255 : 0;
// 
//         parrotOutput[0] = (result == 255) ? 0.9 : 0.1;
// 
// #pragma parrot(output, "imageBinarization", [1]<0.0; 1.0>parrotOutput)

        if(parrotOutput[0] > 0.7)
            result = 255;
        else
            result = 0;

        dst[imageW * iy + ix] = result;
    };
}

extern "C"
void cuda_imageBinarization(
    unsigned char *d_dst,
    int imageW,
    int imageH
)
{
    //printf("cuda image binarization\n");
#pragma parrot.start("imageBinarization")

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    imageBinarization<<<grid, threads>>>(d_dst, imageW, imageH);

    cudaDeviceSynchronize();

#pragma parrot.end("imageBinarization")
}


extern "C"
cudaError_t CUDA_Bind2TextureArray()
{
    return cudaBindTextureToArray(texImage, a_Src);
}

extern "C"
cudaError_t CUDA_UnbindTexture()
{
    return cudaUnbindTexture(texImage);
}

extern "C"
cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    cudaError_t error;

    error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);
    error = cudaMemcpyToArray(a_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                             );

    return error;
}


extern "C"
cudaError_t CUDA_FreeArray()
{
    return cudaFreeArray(a_Src);
}

