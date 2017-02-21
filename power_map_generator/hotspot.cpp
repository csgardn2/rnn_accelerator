/// \file
/// File Name:                      hotspot.cpp \n
/// Date created:                   Fri Feb 17 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or gaussian.h for documentation.

#include <cmath>
#include "hotspot.h"

float hotspot_t::operator()(float x, float y) const
{
    
    float delta_x = x - this->mean_x;
    float delta_y = y - this->mean_y;
    
    return
        this->amplitude
      * std::exp
        (
            -(delta_x * delta_x) / (2 * this->stddev_x * this->stddev_x)
            -(delta_y * delta_y) / (2 * this->stddev_y * this->stddev_y)
        );
    
}

float hotspot_t::calculate_falloff_x(float threshold) const
{
    return std::sqrt
    (
        (-2.0f * this->stddev_x * this->stddev_x)
      * std::log(threshold / this->amplitude)
    );
}

float hotspot_t::calculate_falloff_y(float threshold) const
{
    return std::sqrt
    (
        (-2.0f * this->stddev_y * this->stddev_y)
      * std::log(threshold / this->amplitude)
    );
}

hotspot_stage_t hotspot_t::advance_age()
{
    
    switch (this->stage)
    {
        
        case hotspot_stage_t::GROWING:
        {
            float cur_amplitude = this->amplitude + this->aging_rate;
            if (cur_amplitude >= this->peak_amplitude)
            {
                cur_amplitude = this->peak_amplitude;
                this->stage = hotspot_stage_t::SHRINKING;
            }
            this->amplitude = cur_amplitude;
            break;
        }
        
        case hotspot_stage_t::SHRINKING:
        {
            float cur_amplitude = this->amplitude - this->aging_rate;
            if (cur_amplitude <= 0.0f)
            {
                cur_amplitude = 0.0f;
                this->stage = hotspot_stage_t::DEAD;
            }
            this->amplitude = cur_amplitude;
            break;
        }
        
        default:
            break;
        
    }
    
    return this->stage;
    
}

