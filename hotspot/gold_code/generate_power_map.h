/// \file
/// File Name:                      generate_power_map.h \n
/// Date created:                   Fri Feb 10 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.
 
#ifndef HEADER_GUARD_POWER_MAP_GENERATOR
#define HEADER_GUARD_POWER_MAP_GENERATOR

#include "power_map_state.h"

/// \brief Generate a random power map by superimposing a bunch of hills
/// of random sizes at random (x, y) locations which decay at a random rate.
void generate_power_map
(
    
    /// [inout] Used to track where each hotspot on the power map is.  The power
    /// map state is then advanced slightly and written back in-place.
    power_map_state_t* power_map_state,
    
    /// [out] 2D array that is overwritten with a new heat map.
    float* power_map
    
);

#endif // header guard

