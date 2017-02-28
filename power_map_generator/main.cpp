/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Fri Feb 10 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.

#include <iostream>
#include <random>
#include <string>

#include "args.h"
#include "generate_power_map.h"
#include "png.h"
#include "power_map_state.h"

/// This main function is mostly for testing / tutorial purposes.
/// See args.h for command line usage
/// power_map_generator is where the real action is
int main(int argc, char** argv)
{
    
    args_t args;
    arg_error_t argument_parsing_status = args.parse(argc, argv);
    if (argument_parsing_status != arg_error_code_t::SUCCESS)
    {
        std::cerr << argument_parsing_status.error_string << "\"\n";
        return -1;
    }
    
    std::cout << "width = " << args.width << '\n'
        << "height = " << args.height << '\n'
        << "time_steps = " << args.time_steps << '\n'
        << "base_filename = \"" << args.base_filename << "\"\n";
    
    return 0;
    
    
    power_map_state_t power_map_state
    (
        args.width,
        args.height,
        1024,               // Max hotspots
        10.0f,              // Min peak amplitude
        50.0f,              // Max peak amplitude
        5.0f,               // Min stdev
        15.0f,              // Max stddev
        5.0f,               // Min aging rate
        20.0f               // Max aging rate
    );
    
    png_t encoder(true, args.width, args.height);
    
    unsigned flat_size = args.width * args.height;
    float* power_map = new float[flat_size];
    
//    unsigned char* color_out = encoder.get_grey();
    std::string filename;
    for (unsigned iy = 0, bound = args.time_steps; iy < bound; iy++)
    {
        
        // Compute the power map
        generate_power_map(&power_map_state, power_map);
        
        // Dump a CSV of the power map
        
        
        // Dump a PNG from the power map
        /*
        for (unsigned ix = 0; ix < flat_size; ix++)
        {
            signed cur_color = signed(power_map[ix]);
            if (cur_color < 0)
                cur_color = 0;
            if (cur_color > 255)
                cur_color = 255;
            color_out[ix] = cur_color;
        }
        filename = args.base_filename;
        filename += std::to_string(iy);
        filename += ".png";
        encoder.write_to_file(filename.c_str());
        */
        
    }
    
    delete[] power_map;
    
    return 0;
    
}

