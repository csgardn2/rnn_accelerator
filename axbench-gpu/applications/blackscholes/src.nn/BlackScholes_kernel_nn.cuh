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



///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
__device__ inline float cndGPU(float d)
{
    const float       A1 = 0.31938153f;
    const float       A2 = -0.356563782f;
    const float       A3 = 1.781477937f;
    const float       A4 = -1.821255978f;
    const float       A5 = 1.330274429f;
    const float RSQRT2PI = 0.39894228040143267793994605993438f;

    float
    K = 1.0f / (1.0f + 0.2316419f * fabsf(d));

    float
    cnd = RSQRT2PI * __expf(- 0.5f * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0f - cnd;

    return cnd;
}


///////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for both call and put
///////////////////////////////////////////////////////////////////////////////
__device__ inline void BlackScholesBodyGPU(
    float &CallResult,
    float &PutResult,
    float S, //Stock price
    float X, //Option strike
    float T, //Option years
    float R, //Riskless rate
    float V  //Volatility rate
)
{
    float sqrtT, expRT;
    float d1, d2, CNDD1, CNDD2;

    float parrotInput[3];
    float parrotOutput[1];

    parrotInput[0] = S;
    parrotInput[1] = X;
    parrotInput[2] = T;

float layer_1_0 = parrotInput[0] * -1.517956 + parrotInput[1] * -23.738037 + parrotInput[2] * 48.788017 + 1.0f * -1.904377;

float layer_1_1 = parrotInput[0] * 0.941893 + parrotInput[1] * -22.499035 + parrotInput[2] * -4.052216 + 1.0f * 0.340915;

float layer_1_2 = parrotInput[0] * 1.737908 + parrotInput[1] * -15.498505 + parrotInput[2] * -1.600299 + 1.0f * 0.407830;

float layer_1_3 = parrotInput[0] * -3.457692 + parrotInput[1] * 3.864713 + parrotInput[2] * -1.743620 + 1.0f * -0.312864;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * -0.890064 + sigmoid(layer_1_1, 0.500000) * -2.869173 + sigmoid(layer_1_2, 0.500000) * -1.481759 + sigmoid(layer_1_3, 0.500000) * 6.760577 + 1.0f * -1.203726;

layer_2_0 = sigmoid(layer_2_0, 0.5);

float layer_2_1 = sigmoid(layer_1_0, 0.500000) * -0.610240 + sigmoid(layer_1_1, 0.500000) * -1.277162 + sigmoid(layer_1_2, 0.500000) * -4.149025 + sigmoid(layer_1_3, 0.500000) * 12.601338 + 1.0f * -0.964733;

layer_2_1 = sigmoid(layer_2_1, 0.5);

float layer_2_2 = sigmoid(layer_1_0, 0.500000) * 0.032721 + sigmoid(layer_1_1, 0.500000) * -2.250315 + sigmoid(layer_1_2, 0.500000) * -1.582945 + sigmoid(layer_1_3, 0.500000) * 31.708364 + 1.0f * -1.008929;

layer_2_2 = sigmoid(layer_2_2, 0.5);

float layer_2_3 = sigmoid(layer_1_0, 0.500000) * 0.479379 + sigmoid(layer_1_1, 0.500000) * -2.225986 + sigmoid(layer_1_2, 0.500000) * -0.474371 + sigmoid(layer_1_3, 0.500000) * 168.709946 + 1.0f * -2.223274;

layer_2_3 = sigmoid(layer_2_3, 0.5);

float layer_3_0 = sigmoid(layer_2_0, 0.500000) * -2.133898 + sigmoid(layer_2_1, 0.500000) * -2.628783 + sigmoid(layer_2_2, 0.500000) * -1.735862 + sigmoid(layer_2_3, 0.500000) * -0.773615 + 1.0f * 0.340383;

layer_3_0 = sigmoid(layer_3_0, 0.5);

parrotOutput[0] = layer_3_0;

// parrotOutput[0] = layer_3_0;
//     sqrtT = sqrtf(T);
//     d1 = (__logf(S / X) + (R + 0.5f * V * V) * T) / (V * sqrtT);
//     d2 = d1 - V * sqrtT;
// 
//     CNDD1 = cndGPU(d1);
//     CNDD2 = cndGPU(d2);
// 
//     //Calculate Call and Put simultaneously
//     expRT = __expf(- R * T);
//     CallResult = S * CNDD1 - X * expRT * CNDD2;
//     parrotOutput[0] = CallResult;
// #pragma parrot(output, "BlackScholesBodyGPU", [1]<0.0; 1.0>parrotOutput)

    CallResult = parrotOutput[0];
    PutResult  = X * expRT * (1.0f - CNDD2) - S * (1.0f - CNDD1);
}


////////////////////////////////////////////////////////////////////////////////
//Process an array of optN options on GPU
////////////////////////////////////////////////////////////////////////////////
__global__ void BlackScholesGPU(
    float *d_CallResult,
    float *d_PutResult,
    float *d_StockPrice,
    float *d_OptionStrike,
    float *d_OptionYears,
    float Riskfree,
    float Volatility,
    int optN
)
{
    ////Thread index
    //const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
    ////Total number of threads in execution grid
    //const int THREAD_N = blockDim.x * gridDim.x;

    const int opt = blockDim.x * blockIdx.x + threadIdx.x;

    //No matter how small is execution grid or how large OptN is,
    //exactly OptN indices will be processed with perfect memory coalescing
    //for (int opt = tid; opt < optN; opt += THREAD_N)
    if (opt < optN)
        BlackScholesBodyGPU(
            d_CallResult[opt],
            d_PutResult[opt],
            d_StockPrice[opt],
            d_OptionStrike[opt],
            d_OptionYears[opt],
            Riskfree,
            Volatility
        );
}
