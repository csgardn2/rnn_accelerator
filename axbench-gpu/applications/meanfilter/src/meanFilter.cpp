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
// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
  #include <GLUT/glut.h>
  #ifndef glutCloseFunc
  #define glutCloseFunc glutWMCloseFunc
  #endif
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "meanFilter_kernels.h"

#include <stdio.h>

// includes, project
#include <helper_functions.h> // includes for SDK helper functions
#include <helper_cuda.h>      // includes for cuda initialization and error checking

const char *filterMode[] =
{
    "No Filtering",
    "Mean Texture",
    "Mean SMEM+Texture",
    NULL
};

//
// Cuda example code that implements the Mean edge detection
// filter. This code works for 8-bit monochrome images.
//
// Use the '-' and '=' keys to change the scale factor.
//
// Other keys:
// I: display image
// T: display Mean edge detection (computed solely with texture)
// S: display Mean edge detection (computed with texture and shared memory)

void cleanup(void);
void initializeData(char *file) ;

#define MAX_EPSILON_ERROR 5.0f
#define REFRESH_DELAY     10 //ms

const char *sSDKsample = "CUDA Mean Edge-Detection";

static int wWidth   = 512; // Window width
static int wHeight  = 512; // Window height
static int imWidth  = 0;   // Image width
static int imHeight = 0;   // Image height

// Code to handle Auto verification
const int frameCheckNumber = 4;
int fpsCount = 0;      // FPS count for averaging
int fpsLimit = 8;      // FPS limit for sampling
unsigned int frameCount = 0;
unsigned int g_TotalErrors = 0;
StopWatchInterface *timer = NULL;
unsigned int g_Bpp;
unsigned int g_Index = 0;

bool g_bQAReadback = false;

// Display Data
static GLuint pbo_buffer = 0;  // Front and back CA buffers
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

static GLuint texid = 0;       // Texture for display
unsigned char *pixels = NULL;  // Image pixel data on the host
float imageScale = 1.f;        // Image exposure
enum MeanDisplayMode g_MeanDisplayMode;

int *pArgc   = NULL;
char **pArgv = NULL;

extern "C" void runAutoTest(int argc, char **argv);

#define OFFSET(i) ((char *)NULL + (i))
#define MAX(a,b) ((a > b) ? a : b)

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

        g_Bpp = 1;
    }
    else if (!strcmp(&file[file_length-3], "ppm"))
    {
        if (sdkLoadPPM4(file, &pixels, &w, &h) != true)
        {
            printf("Failed to load PPM image file: %s\n", file);
            exit(EXIT_FAILURE);
        }

        g_Bpp = 4;
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

    imWidth = (int)w;
    imHeight = (int)h;
    setupTexture(imWidth, imHeight, pixels, g_Bpp);

    memset(pixels, 0x0, g_Bpp * sizeof(Pixel) * imWidth * imHeight);

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

    Pixel *d_result;
    checkCudaErrors(cudaMalloc((void **)&d_result, imWidth*imHeight*sizeof(Pixel)));

    g_MeanDisplayMode = MeanDISPLAY_MeanTEX;

    MeanFilter(d_result, imWidth, imHeight, g_MeanDisplayMode, imageScale);
    checkCudaErrors(cudaDeviceSynchronize());

    unsigned char *h_result = (unsigned char *)malloc(imWidth*imHeight*sizeof(Pixel));
    checkCudaErrors(cudaMemcpy(h_result, d_result, imWidth*imHeight*sizeof(Pixel), cudaMemcpyDeviceToHost));
    sdkSavePGM(dump_file, h_result, imWidth, imHeight);

    checkCudaErrors(cudaFree(d_result));
    free(h_result);

    exit(EXIT_SUCCESS);
}

int main(int argc, char **argv)
{

    pArgc = &argc;
    pArgv = argv;

    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("\nUsage: MeanFilter <options>\n");
        printf("\t\t-mode=n (0=original, 1=texture, 2=smem + texture)\n");
        printf("\t\t-file=ref_orig.pgm (ref_tex.pgm, ref_shared.pgm)\n\n");
        exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "file"))
    {
        g_bQAReadback = true;
        runAutoTest(argc, argv);
    }

    
}
