/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Wed Nov 9 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 -lpng\n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 14.04 \n
/// Target architecture:            x86 64-bit \n */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "png.h"

void gradient
(
    unsigned width,
    unsigned height,
    const uint8_t* red_in,
    const uint8_t* green_in,
    const uint8_t* blue_in,
    uint8_t* red_out,
    uint8_t* green_out,
    uint8_t* blue_out
);

int main(int argc, char** argv)
{
    
    // Argument sanity check
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " [input.png] [output.png]\n";
        return -1;
    }
    
    // Open input image and load into memory
    png_t decoder;
    if (SUCCESS != decoder.read_from_file(argv[1]))
    {
        std::cerr << "Error.  Could not open input image \"" << argv[1] << "\" for reading\n";
        return -1;
    }
    
    unsigned width = decoder.get_width();
    unsigned height = decoder.get_height();
    
    // Allocate output image
    png_t encoder(false, width, height);
    
    // Copy input image to output for testing purposes
    gradient
    (
        width,
        height,
        decoder.get_red(),
        decoder.get_green(),
        decoder.get_blue(),
        encoder.get_red(),
        encoder.get_green(),
        encoder.get_blue()
    );
    
    // Write output image to file
    if (SUCCESS != encoder.write_to_file(argv[2]))
    {
        std::cerr << "Error.  Could not open output image \"" << argv[2] << "\" for writing\n";
        return -1;
    }
    
    return 0;
    
}

