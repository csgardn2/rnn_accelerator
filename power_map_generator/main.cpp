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
    
    bool* consumed = new bool[argc];
    args_t args;
    parsing_status_t parsing_status = args.parse(argc, argv, consumed);
    if (parsing_status != parsing_status_t::SUCCESS)
    {
        std::cerr << "Error.  " << args_t::enum_to_string(parsing_status) << '\n';
        delete[] consumed;
        return -1;
    }
    for (signed ix = 1; ix < argc; ix++)
    {
        if (!consumed[ix])
            std::cerr << "Warning.  Ignoring unrecognized argument \"" << argv[ix] << "\"\n";
    }
    delete[] consumed;
    
    std::cout
        << "width = " << args.width << '\n'
        << "height = " << args.height << '\n'
        << "time_steps = " << args.time_steps << '\n'
        << "base_png_filename = \"" << args.base_png_filename << "\"\n"
        << "base_txt_filename = \"" << args.base_txt_filename << "\"\n"
        << "max_hotspots = " << args.max_hotspots << '\n'
        << "min_peak_amplitude = " << args.min_peak_amplitude << '\n'
        << "max_peak_amplitude = " << args.max_peak_amplitude << '\n'
        << "min_stddev = " << args.min_stddev << '\n'
        << "max_stddev = " << args.max_stddev << '\n'
        << "min_aging_rate = " << args.min_aging_rate << '\n'
        << "max_aging_rate = " << args.max_aging_rate << '\n';
    
    return 0;
    
    power_map_state_t power_map_state
    (
        args.width,
        args.height,
        args.max_hotspots,
        args.min_peak_amplitude,
        args.max_peak_amplitude,
        args.min_stddev,
        args.max_stddev,
        args.min_aging_rate,
        args.max_aging_rate
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

