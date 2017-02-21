/// \file
/// File Name:                      generate_power_map.cpp \n
/// Date created:                   Fri Feb 10 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or power_map_generator.h for documentation.

#include <iostream>
#include <random>
#include <vector>
#include "hotspot.h"
#include "generate_power_map.h"
#include "power_map_state.h"

void generate_power_map(power_map_state_t* power_map_state, float* power_map)
{
    
    // Advance the ages of all current hotspots on the heat map.  Spawn in new
    // hotspots each time one reaches the end of its lifecycle.
    unsigned num_hotspots = power_map_state->hotspots.size();
    for (unsigned ix = 0; ix < num_hotspots; ix++)
    {
        if (power_map_state->hotspots[ix].advance_age() == hotspot_stage_t::DEAD)
            power_map_state->overwrite_random_hotspot(ix);
    }
    
    // Give the simulation a chance to create new hotspots
    power_map_state->add_random_hotspots();
    
    // Zero the power map before drawing hotspots on it.
    unsigned width = power_map_state->width;
    unsigned height = power_map_state->height;
    unsigned flat_size = width * height;
    for (unsigned ix = 0; ix < flat_size; ix++)
        power_map[ix] = 0.0f;
    
    // Render each hotspot onto the 2D power grid
    float threshold = power_map_state->max_peak_amplitude * 0.01f;
    for (const hotspot_t& cur_hotspot : power_map_state->hotspots)
    {
        
        // Calculate a bounding box for drawing the hotspot.
        
        float falloff_y = cur_hotspot.calculate_falloff_y(threshold);
        
        signed north = signed(cur_hotspot.mean_y - falloff_y);
        if (north < 0)
            north = 0;
        else if (north >= signed(height))
            north = height - 1;
        
        signed south = signed(cur_hotspot.mean_y + falloff_y);
        if (south < 0)
            south = 0;
        else if (south > signed(height))
            south = height;
        
        float falloff_x = cur_hotspot.calculate_falloff_x(threshold);
        
        signed west = signed(cur_hotspot.mean_x - falloff_x);
        
        if (west < 0)
            west = 0;
        else if (west >= signed(width))
            west = width - 1;
        signed east = signed(cur_hotspot.mean_x + falloff_x);
        if (east < 0)
            east = 0;
        else if (east > signed(width))
            east = width;
        /*
        std::cout
            << "peak = " << cur_hotspot(cur_hotspot.mean_x, cur_hotspot.mean_y)
            << " north = " << north 
            << " south = " << south
            << " west = " << west
            << " east = " << east
            << '\n';
        */
        // Render the hotspot
        for (signed iy = north; iy < south; iy++)
        {
            float* power_row = power_map + width * iy;
            for (signed ix = west; ix < east; ix++)
            {
                float eval = cur_hotspot(float(ix), float(iy));
                power_row[ix] += eval;
//                std::cout << eval << '\n';
            }
        }
        
    }
    
}

