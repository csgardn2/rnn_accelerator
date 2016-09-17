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

#ifndef __MEANFILTER_KERNELS_H_
#define __MEANFILTER_KERNELS_H_

typedef unsigned char Pixel;

// global determines which filter to invoke
enum MeanDisplayMode
{
    MeanDISPLAY_IMAGE = 0,
    MeanDISPLAY_MeanTEX,
    MeanDISPLAY_MeanSHARED
};


extern enum MeanDisplayMode g_MeanDisplayMode;

extern "C" void MeanFilter(Pixel *odata, int iw, int ih, enum MeanDisplayMode mode, float fScale);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);

#endif

