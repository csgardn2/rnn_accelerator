//------------------------------------------------//
// Amir: Headers
//------------------------------------------------//
// CUDA utilities and system includes
#include <cuda_runtime.h>
// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "CudaFilterKernel.h"
#include <stdio.h>
// includes, project
#include <helper_functions.h> // includes for SDK helper functions

// Main Implementation
#define CLAMP_8bit(x) max(0, min(255, (x)))

const int TILE_WIDTH        = 16;
const int TILE_HEIGHT       = 16;


#if     defined(COMPILE_SOBEL_FILTER)
        const int FILTER_RADIUS = 1;
#elif   defined(COMPILE_LAPLACIAN_FILTER)
        const int FILTER_RADIUS = 1;
#elif   defined(COMPILE_AVERAGE_FILTER)
        const int FILTER_RADIUS = 3;
#elif   defined(COMPILE_HIGH_BOOST_FILTER)
        const int FILTER_RADIUS = 3;
#endif

const int FILTER_DIAMETER   = 2 * FILTER_RADIUS + 1;
const int FILTER_AREA       = FILTER_DIAMETER * FILTER_DIAMETER;


const int BLOCK_WIDTH       = TILE_WIDTH    + 2 * FILTER_RADIUS;
const int BLOCK_HEIGHT      = TILE_HEIGHT   + 2 * FILTER_RADIUS;

const int EDGE_VALUE_THRESHOLD  = 127;
const int HIGH_BOOST_FACTOR     = 10;

/* Device Memory */
Pixel*  d_LumaPixelsIn  = NULL;
Pixel*  d_LumaPixelsOut = NULL;


#ifdef COMPILE_SOBEL_FILTER
    
        const float h_SobelMatrix[9] = {-1,0,1,-2,0,2,-1,0,1};
        float*      d_SobelMatrix;

#endif 


#ifdef COMPILE_LAPLACIAN_FILTER

        const float h_LaplacianMatrix[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
        float*      d_LaplacianMatrix;

#endif


// frame size
int* d_Width    = NULL;
int* d_Height   = NULL;


/* Host Memory */
int     h_Width;
int     h_Height;
long    h_DataLength;


// CUDA Kernels
__global__ void SobelFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height, float* d_SobelMatrix);
__global__ void LaplacianFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height, float* d_LaplacianMatrix);
__global__ void AverageFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height);
__global__ void HighBoostFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height);


bool CUDAInit(int width, int height)
{
        // Later check for CUDA Device
        h_Width     = width;
        h_Height    = height;
        return true;
}

void CUDARelease()
{
        cudaFree(d_LumaPixelsIn);
        cudaFree(d_LumaPixelsOut);
        cudaFree(d_Width);
        cudaFree(d_Height);
}

void FilterWrapper(Pixel* pImageIn)
{
        int gridWidth   = (h_Width + TILE_WIDTH - 1) / TILE_WIDTH;
        int gridHeight  = (h_Height + TILE_HEIGHT - 1) / TILE_HEIGHT;

        dim3 dimGrid(gridWidth, gridHeight);
        dim3 dimBlock(BLOCK_WIDTH, BLOCK_HEIGHT);

#ifdef COMPILE_SOBEL_FILTER
        SobelFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height, d_SobelMatrix);

#elif defined COMPILE_LAPLACIAN_FILTER
        LaplacianFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height, d_LaplacianMatrix);

#elif defined COMPILE_AVERAGE_FILTER
        AverageFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height);

#elif defined COMPILE_HIGH_BOOST_FILTER
        HighBoostFilter<<< dimGrid, dimBlock >>>(d_LumaPixelsIn, d_LumaPixelsOut, d_Width, d_Height);

#endif

        cudaThreadSynchronize();
}

bool CUDABeginDetection(Pixel* pImageIn, long dataLength)
{

        h_DataLength = dataLength;


#ifdef COMPILE_SOBEL_FILTER

        if(d_SobelMatrix == NULL)
        {
                cudaMalloc((void**)&d_SobelMatrix, sizeof(float) * FILTER_AREA);
                cudaMemcpy(d_SobelMatrix, h_SobelMatrix, sizeof(float) * FILTER_AREA, cudaMemcpyHostToDevice);
        }
#endif

#ifdef COMPILE_LAPLACIAN_FILTER
        if(d_LaplacianMatrix == NULL)
        {
                cudaMalloc((void**)&d_LaplacianMatrix, sizeof(float) * FILTER_AREA);
                cudaMemcpy(d_LaplacianMatrix, h_LaplacianMatrix, sizeof(float)* FILTER_AREA, cudaMemcpyHostToDevice);
        }
#endif


        if(d_Width == NULL && d_Height == NULL)
        {
                cudaMalloc(&d_Width, sizeof(int));
                cudaMalloc(&d_Height, sizeof(int));

                cudaMemcpy(d_Width,  &h_Width, sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_Height, &h_Height, sizeof(int), cudaMemcpyHostToDevice); 
        }

        if(d_LumaPixelsIn == NULL)
        {
                cudaMalloc((void**)&d_LumaPixelsIn, sizeof(Pixel) * h_DataLength / 2);
        }

        if(d_LumaPixelsOut == NULL)
        {
                cudaMalloc((void**)&d_LumaPixelsOut, sizeof(Pixel) * h_DataLength / 2);
        }

        cudaMemcpy((void*)d_LumaPixelsIn, (void*)pImageIn, sizeof(Pixel) * h_DataLength / 2, cudaMemcpyHostToDevice);

        FilterWrapper(pImageIn);

        return true;
}

bool CUDAEndDetection(Pixel* pImageOut)
{

#ifdef COMPILE_SOBEL_FILTER
        //memset(pImageOut + h_DataLength / 2, 128, h_DataLength / 2);
#elif defined COMPILE_LAPLACIAN_FILTER
        //memset(pImageOut + h_DataLength / 2, 128, h_DataLength / 2);
#endif

        cudaMemcpy(pImageOut, d_LumaPixelsOut, sizeof(Pixel) * h_DataLength / 2, cudaMemcpyDeviceToHost);

        return true;
}


/**************************************************************/
//                      Sobel Filter                          //
/**************************************************************/
__global__ void SobelFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height, float* d_SobelMatrix)
{
    __shared__ Pixel sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;     //- FILTER_RADIUS;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;    //- FILTER_RADIUS;

    if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
    {
        int index = y * (*width) + x;
        g_DataOut[index] = g_DataIn[index];

        return;
    }

    //No filtering for the edges
//  x = max(FILTER_RADIUS, x);
//  x = min(x, *width  - FILTER_RADIUS - 1);
//  y = max(FILTER_RADIUS, y);
//  y = min(y, *height - FILTER_RADIUS - 1);

    int index = y * (*width) + x;
    int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

    sharedMem[sharedIndex] = g_DataIn[index];

    __syncthreads();

    if(     threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
        &&  threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
    {
        float sumX = 0, sumY=0;

        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
            for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
            {
                float centerPixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
                sumX += centerPixel * d_SobelMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
                sumY += centerPixel * d_SobelMatrix[(dx + FILTER_RADIUS) * FILTER_DIAMETER + (dy+FILTER_RADIUS)];
            }

            g_DataOut[index] = abs(sumX) + abs(sumY) > EDGE_VALUE_THRESHOLD ? 255 : 0;
    }
}

/**************************************************************/
//                      Average Filter                        //
/**************************************************************/
__global__ void AverageFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height)
{
    __shared__ Pixel sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

    if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
    {
        int index = y * (*width) + x;
        g_DataOut[index] = g_DataIn[index];

        return;
    }

    //No filtering for the edges
//  x = max(FILTER_RADIUS, x);
//  x = min(x, *width  - FILTER_RADIUS - 1);
//  y = max(FILTER_RADIUS, y);
//  y = min(y, *height - FILTER_RADIUS - 1);

    int index = y * (*width) + x;
    int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

    sharedMem[sharedIndex] = g_DataIn[index];

    __syncthreads();

    if(     threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
        &&  threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
    {
        float sum = 0;

        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
        for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
        {
            float pixelValue = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
            sum += pixelValue;
        }

        g_DataOut[index] = (Pixel)(sum / FILTER_AREA);
    }   
}


/**************************************************************/
//                      Laplacian Filter                      //
/**************************************************************/
__global__ void LaplacianFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height, float* d_LaplacianMatrix)
{
    // original code
    #if defined (orig_code)
        __shared__ Pixel sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

        int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
        int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

        if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
        {
            int index = y * (*width) + x;
            g_DataOut[index] = g_DataIn[index];

            return;
        }

        int index = y * (*width) + x;
        int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

        sharedMem[sharedIndex] = g_DataIn[index];

        __syncthreads();

        if(     threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
            &&  threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
        {
            float sum = 0;

            for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
                for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
                {
                    float centerPixel = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
                    sum += centerPixel * d_LaplacianMatrix[(dy + FILTER_RADIUS) * FILTER_DIAMETER + (dx+FILTER_RADIUS)];
                }

                //FIXME abs?
                Pixel res = max(0, min((Pixel)sum, 255));
                g_DataOut[index] = res;
        }  
    #endif 
}


/**************************************************************/
//                      High Booset Filter                    //
/**************************************************************/
__global__ void HighBoostFilter(Pixel* g_DataIn, Pixel* g_DataOut, int* width, int* height)
{
    __shared__ Pixel sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

    if( x < FILTER_RADIUS || x > *width  - FILTER_RADIUS - 1 || y < FILTER_RADIUS || y > *height - FILTER_RADIUS - 1)
    {
        int index = y * (*width) + x;
        g_DataOut[index] = g_DataIn[index];

        return;
    }

    //No filtering for the edges
//  x = max(FILTER_RADIUS, x);
//  x = min(x, *width  - FILTER_RADIUS - 1);
//  y = max(FILTER_RADIUS, y);
//  y = min(y, *height - FILTER_RADIUS - 1);

    int index = y * (*width) + x;
    int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

    Pixel centerPixel = sharedMem[sharedIndex] = g_DataIn[index];

    __syncthreads();

    if(     threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
        &&  threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
    {
        float sum = 0;

        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
            for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
            {
                float pixelValue = (float)(sharedMem[sharedIndex + (dy * blockDim.x + dx)]);
                sum += pixelValue;
            }

            g_DataOut[index] = CLAMP_8bit(centerPixel + HIGH_BOOST_FACTOR * (Pixel)(centerPixel - sum / FILTER_AREA));
}

/**************************************************************/
//                      Median Filter                         //
/**************************************************************/
/*__global__ void MedianFilter(BYTE* g_DataIn, BYTE* g_DataOut, int* width, int* height)
{
    __shared__ BYTE sharedMem[BLOCK_HEIGHT*BLOCK_WIDTH];

    int x = blockIdx.x * TILE_WIDTH + threadIdx.x ;//- FILTER_RADIUS;
    int y = blockIdx.y * TILE_HEIGHT + threadIdx.y ;//- FILTER_RADIUS;

    //No filtering for the edges
    x = max(FILTER_RADIUS, x);
    x = min(x, *width  - FILTER_RADIUS - 1);
    y = max(FILTER_RADIUS, y);
    y = min(y, *height - FILTER_RADIUS - 1);

    int index = y * (*width) + x;
    int sharedIndex = threadIdx.y * blockDim.y + threadIdx.x;

    sharedMem[sharedIndex] = g_DataIn[index];

    __syncthreads();

    BYTE sortCuda[256]; for(int i=0;i<256;++i) sortCuda[i]=0;
    
    if(     threadIdx.x >= FILTER_RADIUS && threadIdx.x < BLOCK_WIDTH - FILTER_RADIUS 
        &&  threadIdx.y >= FILTER_RADIUS && threadIdx.y < BLOCK_HEIGHT - FILTER_RADIUS)
    {
        //float sum = 0;

        for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; ++dy)
            for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; ++dx)
            {
                sortCuda[(sharedMem[sharedIndex + (dy * blockDim.x + dx)])] += 1;

                //sum += pixelValue;
            }

            //BYTE res = (BYTE)(sum / FILTER_AREA);

            int cnt=0;
            int res=0;
            for(int i=0; i<256; ++i)
                if(cnt>=127)
                {
                    res = i;
                    break;
                }
                else
                {
                    cnt+=sortCuda[i];
                }

            g_DataOut[index] = res;
    }*/ 
}
// 

void initializeData(char *file) ;

// Display Data
Pixel *pixels = NULL;  // Image pixel data on the host
int imWidth   = 0;
int imHeight  = 0;

int *pArgc   = NULL;
char **pArgv = NULL;

extern "C" void runAutoTest(int argc, char **argv);

void initializeData(char *file)
{
    unsigned int w, h;
    size_t file_length= strlen(file);

    if (!strcmp(&file[file_length-3], "pgm"))
    {
        if (sdkLoadPGM<unsigned char>(file, &pixels, &w, &h) != true)
        {
            printf("Failed to load PGM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
    else if (!strcmp(&file[file_length-3], "ppm"))
    {
        if (sdkLoadPPM4(file, &pixels, &w, &h) != true)
        {
            printf("Failed to load PPM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        // cudaDeviceReset causes the driver to clean up all state. While
        // not mandatory in normal operation, it is good practice.  It is also
        // needed to ensure correct operation when the application is being
        // profiled. Calling cudaDeviceReset causes all profile data to be
        // flushed before the application exits
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    imWidth = w;
    imHeight = h;

    // 1) Initialize the Width and Height of the Image
    CUDAInit(w, h);
    // 2) Initialize the device memory and copy data from host to device
    CUDABeginDetection(pixels, imWidth * imHeight * 2); 
    // 3) Copy data back from device to host
    CUDAEndDetection(pixels);
    // 4) Release the device memory
    CUDARelease();


}

void loadDefaultImage(char *loc_exec, char *filename)
{

    const char *image_filename = filename;
    char *image_path = sdkFindFilePath(image_filename, loc_exec);

    if (image_path == NULL)
    {
        printf("Failed to read image file: <%s>\n", image_filename);
        exit(EXIT_FAILURE);
    }

    initializeData(image_path);
    free(image_path);
}

void runAutoTest(int argc, char *argv[])
{
    char *ref_file = NULL;
    char *dump_file = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &ref_file);
    getCmdLineArgumentString(argc, (const char **)argv, "output", &dump_file);

    loadDefaultImage(argv[0], ref_file);
    
    cudaDeviceSynchronize();

    // Save the result image
    sdkSavePGM(dump_file, pixels, imWidth, imHeight);

    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{
    runAutoTest(argc, argv);
}
