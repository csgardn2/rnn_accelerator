#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "binarization.h"

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
#if __CUDA_ARCH__ < 200     //Compute capability 1.x architectures
#define CUPRINTF cuPrintf
#else                       //Compute capability 2.x architectures
#define CUPRINTF(fmt, ...) printf("[%d, %d]:\t" fmt, \
                                  blockIdx.y*gridDim.x+blockIdx.x,\
                                  threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x,\
                                  __VA_ARGS__)
#endif


////////////////////////////////////////////////////////////////////////////////
// binarization kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void binarization(
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
#pragma parrot(input, "binarization", [3]parrotInput)

        unsigned char mi = __min(r, __min(g, b));
        unsigned char ma = __max(r, __max(g, b));

        result = (((unsigned short) ma + (unsigned short) mi) > THRESHOLD * 2 ) ? 255 : 0;

        parrotOutput[0] = (result == 255) ? 0.9 : 0.1;

#pragma parrot(output, "binarization", [1]<0.0; 1.0>parrotOutput)

        if(parrotOutput[0] > 0.7)
            result = 255;
        else
            result = 0;

        dst[imageW * iy + ix] = result;
    };
}

extern "C"
void cuda_binarization(
    unsigned char *d_dst,
    int imageW,
    int imageH
)
{
#pragma parrot.start("binarization")

    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    binarization<<<grid, threads>>>(d_dst, imageW, imageH);

    cudaDeviceSynchronize();

#pragma parrot.end("binarization")
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

