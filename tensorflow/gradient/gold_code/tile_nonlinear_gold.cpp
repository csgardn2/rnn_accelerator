/// \file
/// File Name:                      tile_nonlinear_gold.cpp \n
/// Date created:                   Tues Dec 6 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n

#include <cstdint>

/// Inefficient method of computing the gradient for a single pixel
uint8_t tile_nonlinear_gold(const uint8_t in[9])
{
    
    signed dx = in[5] - in[3];
    signed dy = in[1] - in[7];
    
    if (dx < 0)
        dx = -dx;
    if (dy < 0)
        dy = -dy;
    
    // Technically this should be sqrt(dx^2 + dy^2) but I'm leaving it like this
    // for efficiency.
    unsigned ret = (dx + dy) / 2;
    return ret;
    
}

