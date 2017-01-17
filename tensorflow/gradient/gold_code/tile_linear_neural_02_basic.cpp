/// \file
/// File Name:                      tile_linear_neural_02_basic.cpp \n
/// Date created:                   Tues Dec 6 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */

#include <cstdint>
#include <iostream>

/// These are fairly big const arrays, so they can make text editors run slowly.
/// Therefore they're declared in outside files.
#include "weights_02_basic_65536x512.cpp"
#include "biases_02_basic_65536x512.cpp"

/// Single layer of neurons where there are 256 output neurons.  One neuron for
/// each possible value of an 8-bit color channel.
uint8_t tile_linear_neural_02_basic(const uint8_t in[9])
{
    
    /// Normalize inputs from [0 to 256) to [0 to 1)
    float in_f[9];
    for (unsigned ix = 0; ix < 9; ix++)
        in_f[ix] = float(in[ix]) / 256.0f;
    
    /// matrix_mul_out = in * weights_02_basic
    /// 1 row
    /// 256 columns
    float matrix_mul_out[256];
    
    for (unsigned write_ix = 0; write_ix < 256; write_ix++)
    {
        float accumulator = 0.0f;
        for (unsigned read_ix = 0; read_ix < 9; read_ix++)
        {
            accumulator += in_f[read_ix] * weights_02_basic[256 * read_ix + write_ix];
        }
        matrix_mul_out[write_ix] = accumulator;
    }
    
    // Add neural biases
    for (unsigned ix = 0; ix < 256; ix++)
    {
        matrix_mul_out[ix] += biases_02_basic[ix];
//        std::cout << matrix_mul_out[ix] << ' ';
    }
    
    // The return value is the index of the neuron with the strongest output
    float max_value = matrix_mul_out[0];
    unsigned max_index = 0;
    for (unsigned ix = 1; ix < 256; ix++)
    {
        float fetch = matrix_mul_out[ix];
        if (max_value < fetch)
        {
            max_value = fetch;
            max_index = ix;
        }
    }
    
    return max_index;
    
}

