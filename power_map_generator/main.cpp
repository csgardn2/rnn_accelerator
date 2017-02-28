/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Fri Feb 10 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.

#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "args.h"
#include "color_picker.h"
#include "generate_power_map.h"
#include "png.h"
#include "power_map_state.h"

/// This main function is mostly for testing / tutorial purposes.
/// See args.h for command line usage
/// power_map_generator is where the real action is
int main(int argc, char** argv)
{
    
    // Parse command line arguments
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
    
    // Initialize a grid which will develop hot patches as time progresses
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
    unsigned flat_size = args.width * args.height;
    float* power_map = new float[flat_size];
    
    // Initialize a PNG writer
    png_t png_encoder(false, args.width, args.height);
    unsigned char* red_out = png_encoder.get_red();
    unsigned char* green_out = png_encoder.get_green();
    unsigned char* blue_out = png_encoder.get_blue();
    
    // Compute each iteration of the power map, and dump to the requested files
    bool enable_png_write = args.base_png_filename_status == arg_status_t::FOUND;
    bool enable_txt_write = args.base_txt_filename_status == arg_status_t::FOUND;
    std::string filename;
    unsigned previous_progress = 0;
    float png_white_threshold = args.max_peak_amplitude * 6.0f; // Empirical - Feel free to tweak.
    for (unsigned iy = 0, bound = args.time_steps; iy < bound; iy++)
    {
        
        // Compute the next iteration of the power map.
        // This will only cause a slight change from the previous power map.
        generate_power_map(&power_map_state, power_map);
        
        // Dump a PNG from the power map if requested
        if (enable_png_write)
        {
            
            for (unsigned ix = 0; ix < flat_size; ix++)
            {
                color_t cur_color = pick_color(power_map[ix], 0.0f, png_white_threshold);
                red_out[ix] = cur_color.red;
                green_out[ix] = cur_color.green;
                blue_out[ix] = cur_color.blue;
            }
            
            filename = args.base_png_filename;
            filename += std::to_string(iy);
            filename += ".png";
            
            png_encoder.write_to_file(filename.c_str());
        }
        
        // Dump a TXT of the power map if requested
        if (enable_txt_write)
        {
            
            filename = args.base_txt_filename;
            filename += std::to_string(iy);
            filename += ".txt";
            
            std::ofstream txt_encoder(filename);
            
            txt_encoder << flat_size << '\n';
            
            for (unsigned ix = 0; ix < flat_size; ix++)
                txt_encoder << power_map[ix] << '\n';
            
        }
        
        // Print progress indicator
        unsigned progress = iy * 64 / bound;
        if (progress > previous_progress)
        {
            for (unsigned ix = 0; ix < progress; ix++)
                std::cout << '#';
            for (unsigned ix = progress; ix < 63; ix++)
                std::cout << '-';
            std::cout
                << "\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r"
                   "\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r"
                   "\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r"
                   "\r\r\r\r\r\r\r\r\r\r\r\r\r\r\r";
            std::cout << std::flush;
            previous_progress = progress;
        }
        
    }
    
    std::cout << '\n';
    
    delete[] power_map;
    
    return 0;
    
}

