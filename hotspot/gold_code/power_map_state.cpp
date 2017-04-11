/// \file
/// File Name:                      power_map_state.cpp \n
/// Date created:                   Mon Feb 20 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or power_map_state.h for documentation.

#include <random>
#include <vector>
#include "hotspot.h"
#include "power_map_state.h"

unsigned power_map_state_t::add_random_hotspots()
{
    
    // Calculate the number of new hotspots to add.  This is fairly arbitraty.
    // Here, we fill up half the available space.
    unsigned old_size = this->hotspots.size();
    signed num_new_hotspots = (1 + this->max_hotspots - old_size) / 2;
    if (num_new_hotspots <= 0)
        return 0;
    unsigned new_size = old_size + num_new_hotspots;
    
    // Generate a new random hotspot for each new position we've just allocated.
    this->hotspots.resize(new_size);
    for (unsigned ix = old_size; ix < new_size; ix++)
        this->overwrite_random_hotspot(ix);
    
    return unsigned(num_new_hotspots);
    
}

void power_map_state_t::overwrite_random_hotspot(unsigned ix)
{
    
    hotspot_t* cur_hotspot = &(this->hotspots[ix]);
    
    // Overwrite the new hotspot's amplitude, position, and standard deviation.
    cur_hotspot->amplitude = 0.0f;
    std::uniform_int_distribution<unsigned> mean_x_distribution(0, this->width);
    cur_hotspot->mean_x = mean_x_distribution(this->generator);
    std::uniform_int_distribution<unsigned> mean_y_distribution(0, this->height);
    cur_hotspot->mean_y = mean_y_distribution(this->generator);
    std::uniform_real_distribution<float> stddev_distribution
    (
        this->min_stddev,
        this->max_stddev
    );
    cur_hotspot->stddev_x = stddev_distribution(this->generator);
    cur_hotspot->stddev_y = stddev_distribution(this->generator);
    
    // Start the new hotspot at the begining of its lifecycle.
    cur_hotspot->stage = hotspot_stage_t::GROWING;
    
    // Overwrite peak amplitude: The lifetime of the new item.
    std::uniform_real_distribution<float> peak_amplitude_distribution
    (
        this->min_peak_amplitude,
        this->max_peak_amplitude
    );
    cur_hotspot->peak_amplitude = peak_amplitude_distribution(this->generator);
    
    // Overwrite aging rate
    std::uniform_real_distribution<float> aging_rate_distribution
    (
        this->min_aging_rate,
        this->max_aging_rate
    );
    cur_hotspot->aging_rate = aging_rate_distribution(this->generator);
    
}

