/// \file
/// File Name:                      fourier.cpp \n
/// Date created:                   Wed April 12 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or fourier.h for documentation.

#include <cmath>
#include <iostream>
#include <vector>

float fourier_cos
(
    /// [in] Evaluate a function over one period at intervals of dt.
    const std::vector<float>& samples,
    
    /// [in] The frequency component which you want to find the
    /// amplitude.
    float frequency
){
    
    unsigned num_samples = samples.size();
    float dt = 1.0f / float(num_samples);
    float accumulator = 0.0f;
    float cur_time = 0.0f;
    for (unsigned ix = 0; ix < num_samples; ix++)
    {
        accumulator += samples[ix] * std::cos(2.0f * M_PI * frequency * cur_time);
        cur_time += dt;
    }
    
    return 2.0f * accumulator * dt;
    
}
