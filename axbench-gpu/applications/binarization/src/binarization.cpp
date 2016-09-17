// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "binarization.h"

// includes, project
#include <helper_functions.h> // includes for helper utility functions
#include <helper_cuda.h>      // includes for cuda error checking and initialization

const char *sSDKsample = "CUDA binarization";

const char *filterMode[] =
{
    "Passthrough",
    "KNN method",
    "NLM method",
    "Quick NLM(NLM2) method",
    NULL
};

// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
    "image_passthru.ppm",
    "image_knn.ppm",
    "image_nlm.ppm",
    "image_nlm2.ppm",
    NULL
};

const char *sReference[] =
{
    "ref_passthru.ppm",
    "ref_knn.ppm",
    "ref_nlm.ppm",
    "ref_nlm2.ppm",
    NULL
};

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange
//Source image on the host side
uchar4 *h_Src;
int imageW, imageH;
GLuint shader;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int  g_Kernel = 0;
bool    g_FPS = false;
bool   g_Diag = false;
StopWatchInterface *timer = NULL;

//Algorithms global parameters
const float noiseStep = 0.025f;
const float  lerpStep = 0.025f;
static float knnNoise = 0.32f;
static float nlmNoise = 1.45f;
static float    lerpC = 0.2f;


unsigned int g_TotalErrors = 0;
int *pArgc   = NULL;
char **pArgv = NULL;

void runImageFilters(unsigned char *d_dst)
{
    cuda_binarization(d_dst, imageW, imageH);
}


// shader for displaying floating-point texture
static const char *shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

void runAutoTest(int argc, char **argv)
{


    LoadBMPFile(&h_Src, &imageW, &imageH, argv[1]);

    (CUDA_MallocArray(&h_Src, imageW, imageH));


    unsigned char *d_dst = NULL;
    unsigned char *h_dst = NULL;
    (cudaMalloc((void **)&d_dst, imageW*imageH*sizeof(unsigned char)));
    h_dst = (unsigned char *)malloc(imageH*imageW);

    {
        (CUDA_Bind2TextureArray());
        runImageFilters(d_dst);
        (CUDA_UnbindTexture());
        (cudaDeviceSynchronize());
        (cudaMemcpy(h_dst, d_dst, imageW*imageH*sizeof(unsigned char), cudaMemcpyDeviceToHost));
        sdkSavePGM(argv[2], h_dst, imageW, imageH);
    }

    (CUDA_FreeArray());
    free(h_Src);

    (cudaFree(d_dst));
    free(h_dst);

    // flushed before the application exits
    cudaDeviceReset();
    exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
}


int main(int argc, char **argv)
{
    char *dump_file = NULL;

    pArgc = &argc;
    pArgv = argv;

    runAutoTest(argc, argv);

    cudaDeviceReset();
    exit(EXIT_SUCCESS);
}
