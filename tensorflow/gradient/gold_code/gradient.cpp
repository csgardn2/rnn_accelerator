/// \file
/// File Name:                      gradient.cpp \n
/// Date created:                   Wed Nov 9 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 14.04 \n
/// Target architecture:            x86 64-bit \n

#include <cstdint>
#include <cstring>
#include <iostream>

#include "prototypes.h"

#define DUMP_OUTPUTS 0
#define DUMP_INPUTS 0

/// Applies a per-channel gradient filter such that each pixel is the sum of
/// the x and y differences across that pixel.  To avoid annoyances, the border
/// pixels are set to black.
void gradient
(
    
    /// [in] The number of pixels across a row of the image
    unsigned width,
    
    /// [in] The number of pixels down a single column of the image
    unsigned height,
    
    /// [in] Red color channel of the input image stored in row-major order with
    /// absolutly no padding.
    const uint8_t* red_in,
    
    /// [in] Green input channel
    const uint8_t* green_in,
    
    /// [in] Blue input channel
    const uint8_t* blue_in,
    
    /// [out] Red output channel
    uint8_t* red_out,
    
    /// [out] Green output channel
    uint8_t* green_out,
    
    /// [out] Blue output channel
    uint8_t* blue_out
    
){
    
    // Set north row to black
    std::memset(red_out, 0, width);
    std::memset(green_out, 0, width);
    std::memset(blue_out, 0, width);
    
    // Set south row to black
    unsigned bottom = width * (height - 1);
    std::memset(red_out + bottom, 0, width);
    std::memset(green_out + bottom, 0, width);
    std::memset(blue_out + bottom, 0, width);
    
    // Set east and west columns to black
    unsigned pixels = width * height;
    for
    (
        unsigned east = 0, west = width - 1;
        east < pixels;
        east += width, west += width
    ){
        
        red_out[east] = 0;
        green_out[east] = 0;
        blue_out[east] = 0;
        
        red_out[west] = 0;
        green_out[west] = 0;
        blue_out[west] = 0;
        
    }
    
    #if DUMP_INPUTS
        std::cout << "input_tiles = [\n";
    #endif
    #if DUMP_OUTPUTS
        std::cout << "output_pixels = [";
    #endif
    
    // Perform gradient in a way which is inefficient but is easy to represent
    // as a neural net
    unsigned bound_x = width - 1;
    unsigned bound_y = height - 1;
    for (unsigned iy = 1; iy < bound_y; iy++)
    {
        unsigned row_above = (iy - 1) * width;
        unsigned row_mid = row_above + width;
        unsigned row_below = row_above + 2 * width;
        for (unsigned ix = 1; ix < bound_x; ix++)
        {
            
            uint8_t input_tile[9];
            
            // Collect red channel
            input_tile[0] = red_in[row_above + ix - 1];
            input_tile[1] = red_in[row_above + ix];
            input_tile[2] = red_in[row_above + ix + 1];
            input_tile[3] = red_in[row_mid + ix - 1];
            input_tile[4] = red_in[row_mid + ix];
            input_tile[5] = red_in[row_mid + ix + 1];
            input_tile[6] = red_in[row_below + ix - 1];
            input_tile[7] = red_in[row_below + ix];
            input_tile[8] = red_in[row_below + ix + 1];
            
            #if DUMP_INPUTS
                std::cout << "[";
                for (unsigned iz = 0; iz < 8; iz++)
                    std::cout << unsigned(input_tile[iz]) << ", ";
                std::cout << unsigned(input_tile[8]) << "],\n";
            #endif
            
            // Compute red channel
            uint8_t cur_red_out = tile_linear_neural_01_sucky(input_tile);
            red_out[row_mid + ix] = cur_red_out;
            
            #if DUMP_OUTPUTS
                std::cout << unsigned(cur_red_out) << ", ";
            #endif
            
            // Collect green channel
            input_tile[0] = green_in[row_above + ix - 1];
            input_tile[1] = green_in[row_above + ix];
            input_tile[2] = green_in[row_above + ix + 1];
            input_tile[3] = green_in[row_mid + ix - 1];
            input_tile[4] = green_in[row_mid + ix];
            input_tile[5] = green_in[row_mid + ix + 1];
            input_tile[6] = green_in[row_below + ix - 1];
            input_tile[7] = green_in[row_below + ix];
            input_tile[8] = green_in[row_below + ix + 1];
            
            #if DUMP_INPUTS
                std::cout << "[";
                for (unsigned iz = 0; iz < 8; iz++)
                    std::cout << unsigned(input_tile[iz]) << ", ";
                std::cout << unsigned(input_tile[8]) << "],\n";
            #endif
            
            // Compute green channel
            uint8_t cur_green_out = tile_linear_neural_01_sucky(input_tile);
            green_out[row_mid + ix] = cur_green_out;
            
            #if DUMP_OUTPUTS
                std::cout << unsigned(cur_green_out) << ", ";
            #endif
            
            // Collect blue channel
            input_tile[0] = blue_in[row_above + ix - 1];
            input_tile[1] = blue_in[row_above + ix];
            input_tile[2] = blue_in[row_above + ix + 1];
            input_tile[3] = blue_in[row_mid + ix - 1];
            input_tile[4] = blue_in[row_mid + ix];
            input_tile[5] = blue_in[row_mid + ix + 1];
            input_tile[6] = blue_in[row_below + ix - 1];
            input_tile[7] = blue_in[row_below + ix];
            input_tile[8] = blue_in[row_below + ix + 1];
            
            #if DUMP_INPUTS
                std::cout << "[";
                for (unsigned iz = 0; iz < 8; iz++)
                    std::cout << unsigned(input_tile[iz]) << ", ";
                std::cout << unsigned(input_tile[8]) << "],\n";
            #endif
            
            // Compute blue channel
            uint8_t cur_blue_out = tile_linear_neural_01_sucky(input_tile);
            blue_out[row_mid + ix] = cur_blue_out;
            #if DUMP_OUTPUTS
                std::cout << unsigned(cur_blue_out) << ", ";
            #endif
            
        }
        
    }
    
    #if DUMP_INPUTS
        std::cout << "]\n";
    #endif
    
    #if DUMP_OUTPUTS
        std::cout << "]\n";
    #endif
    
}

