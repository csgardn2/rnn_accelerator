/// \file
/// File Name:                      tile_linear_gold.cpp \n
/// Date created:                   Tues Dec 6 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */

#include <cstdint>

// This gradient function is designed such that it will be easy to approximate
// using a single neural layer.
uint8_t tile_linear_gold(const uint8_t in[9])
{
    
    // We really only need 4 of the 9 input
    // tiles, but the extra tiles are there
    // to ensure that the neural net is
    // capable of zeroing-out noise inputs
    signed dx = in[5] - in[3];
    signed dy = in[1] - in[7];
    
    // Technically this should be
    // sqrt(dx^2 + dy^2) but I'm leaving it
    // like this for efficiency.
    signed ret = (dx + dy) / 2 + 128;
    
    /// Well... almost linear.  It still
    // clamps the final output to [0 to 255]
    /// but this should rarely happen anyway.
    if (ret < 0)
        ret = 0;
    if (ret > 255)
        ret = 255;
    
    return uint8_t(ret);
    
}

