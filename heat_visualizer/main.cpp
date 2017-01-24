/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Tues Jan 17 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "png.h"
#include "utility.h"

/// Command line usage:
///     ./heat_visualizer [temperature_file] [output.png] [width] [height]
/// If width and height are omitted, then the program will attempt to
/// auto-detect the dimensions by assuming that the input temperature file
/// is square.  If the number of entries in the input file are not a perfect
/// square, the program fails.
int main(int argc, char** argv)
{
    
    // Argument check
    if (argc != 5 && argc != 3)
    {
        std::cerr
            << "Error.  "
            << argv[0]
            << " takes 2 or 4 arguments\nUsage: "
            << argv[0]
            << " <temperature_file> <output.png> [min_heat] [max_heat]\n";
        return -1;
    }
    
    // Open input temperature file and copy it to RAM
    std::ifstream input_file(argv[1]);
    if (!input_file.good())
    {
        std::cerr
            << "Error.  Could not open temperature file \""
            << argv[1]
            << "\" for reading\n";
        return -1;
    }
    std::vector<float> heat_map;
    while (true)
    {
        float fetch;
        input_file >> fetch;
        if (!input_file.good())
            break;
        heat_map.push_back(fetch);
    }
    
    // The temperature output files are stupidly different than the input ones
    // in that the outputs are numbererd.  Try to auto-detect numbers and
    // discard them
    unsigned linear_size = heat_map.size();
    if
    (
        linear_size >= 10
     && soft_eq(heat_map[0], 0.0f)
     && soft_eq(heat_map[2], 1.0f)
     && soft_eq(heat_map[4], 2.0f)
     && soft_eq(heat_map[6], 3.0f)
     && soft_eq(heat_map[8], 4.0f)
    ){
        
        unsigned new_linear_size = linear_size / 2;
        for (unsigned ix = 0; ix < new_linear_size; ix++)
            heat_map[ix] = heat_map[ix * 2 + 1];
        
        linear_size = new_linear_size;
        heat_map.resize(new_linear_size);
        
        std::cout
            << "Auto-detected numbering in temperature file \""
            << argv[1]
            << "\".  Ignoring numbering.\n";
        
    }
    
    // Try to infer heat map dimensions from the input file size
    unsigned side_length = std::sqrt(linear_size);
    if ((side_length + 1) * (side_length + 1) == linear_size)
    {
        // Tolerate floating-point rounding errors
        side_length++;
    } else if (side_length > 0 && (side_length - 1) * (side_length - 1) == linear_size) {
        // Tolerate floating-point rounding errors
        side_length--;
    } else if (side_length * side_length != linear_size) {
        std::cerr
            << "Error.  Could not auto-detect heat map dimensions because the "
            << linear_size
            << " elements in \""
            << argv[1]
            << "\" are not a perfect square.\n";
        return -1;
    }
    
    std::cout
        << "Auto-detected heat map dimensions "
        << side_length
        << " x "
        << side_length
        << ".\n";
    
    // Generate the output image for the heat map
    float min_temperature;
    float max_temperature;
    
    if (argc == 5)
    {
        
        min_temperature = atof(argv[3]);
        max_temperature = atof(argv[4]);
        
        if (std::isnan(min_temperature))
        {
            std::cerr
                << "Error.  Min temperature \""
                << argv[3]
                << "\" is not a valid floating point value";
            return -1;
        }
        if (std::isnan(max_temperature))
        {
            std::cerr
                << "Error.  Max temperature \""
                << argv[4]
                << "\" is not a valid floating point value";
            return -1;
        }
        if (min_temperature >= max_temperature)
        {
            std::cerr << "Error.  Max temperature must be greater than min temperature\n";
            return -1;
        }
        
    } else {
        
        min_temperature = min(heat_map);
        max_temperature = max(heat_map);
        
        std::cout
            << "Auto-detected min temperature (black) = " << min_temperature << '\n'
            << "Auto-detected min temperature (white) = " << max_temperature << '\n';
        
    }
    
    png_t encoder(false, side_length, side_length);
    for (unsigned iy = 0; iy < side_length; iy++)
    {
        
        unsigned offset = iy * side_length;
        const float* temperature_row = heat_map.data() + offset;
        unsigned char* red_row = encoder.get_red() + offset;
        unsigned char* green_row = encoder.get_green() + offset;
        unsigned char* blue_row = encoder.get_blue() + offset;
        for (unsigned ix = 0; ix < side_length; ix++)
        {
            
            color_t cur_pixel = pick_color(temperature_row[ix], min_temperature, max_temperature);
            red_row[ix] = cur_pixel.red;
            green_row[ix] = cur_pixel.green;
            blue_row[ix] = cur_pixel.blue;
            
        }
        
    }
    
    if (encoder.write_to_file(argv[2]) != SUCCESS)
    {
        std::cerr
            << "Error.  Failed to encode PNG output file \""
            << argv[2]
            << "\"\n";
        return -1;
    }
    
    return 0;
    
}

