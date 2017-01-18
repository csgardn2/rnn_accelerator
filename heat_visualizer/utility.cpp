/// \file
/// File Name:                      utility.cpp \n
/// Date created:                   Wed Jan 18 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or for documentation.

#include <cmath>
#include <vector>
#include "utility.h"

/// \brief return the y-value of a point on an isosceles triangle of height 1
/// with the other two points on the x-axis with x-value min and max.
float triangle(float x, float min, float max)
{
    
    float normalized = (x - min) / (max - min);    
    
    if (normalized < 0.0f || normalized > 1.0f)
        return 0.0f;
    else if (normalized < 0.5f)
        return x;
    else
        return 1.0f - normalized;
    
}

/// \brief Do a 5-point linear interpolation from black to blue to green to red
/// to white with black being the coldest (min_temperature) and white being
/// the hottest (max_temperature)
color_t pick_color(float temperature, float min_temperature, float max_temperature)
{
    
    float range = max_temperature - min_temperature;
    float interpolation_endpoint_1 = min_temperature + range * 0.25;
    float interpolation_endpoint_2 = min_temperature + range * 0.5;
    float interpolation_endpoint_3 = min_temperature + range * 0.75;
    float interpolation_endpoint_4 = min_temperature + range * 1.25;
    
    float white =
        triangle(temperature, interpolation_endpoint_3, interpolation_endpoint_4);
    float red =
        triangle(temperature, interpolation_endpoint_2, max_temperature)
      + white;
    float green =
        triangle(temperature, interpolation_endpoint_1, interpolation_endpoint_3)
      + white;
    float blue =
        triangle(temperature, min_temperature, interpolation_endpoint_2)
      + white;
    
    signed cast_red = signed(red * 256.0f);
    signed cast_green = signed(green * 256.0f);
    signed cast_blue = signed(blue * 256.0f);
    
    color_t ret;
    
    if (cast_red < 0)
        ret.red = 0;
    else if (cast_red > 255)
        ret.red = 255;
    else
        ret.red = cast_red;
    
    if (cast_green < 0)
        ret.green = 0;
    else if (cast_green > 255)
        ret.green = 255;
    else
        ret.green = cast_green;
    
    if (cast_blue < 0)
        ret.blue = 0;
    else if (cast_blue > 255)
        ret.blue = 255;
    else
        ret.blue = cast_blue;
    
    return ret;
    
}

/// \brief Return the minimum value from a vector.  Returns NAN if an error
/// occurs.
float min(const std::vector<float>& vec)
{
    
    unsigned num_elements = vec.size();
    if (num_elements == 0)
        return NAN;
    
    float cur_min = vec[0];
    for (unsigned ix = 1; ix < num_elements; ix++)
    {
        float fetch = vec[ix];
        if (cur_min > fetch)
            cur_min = fetch;
    }
    
    return cur_min;
    
}

/// \brief Return the maximum value from a vector.  Return NAN if an error
/// occurs
float max(const std::vector<float>& vec)
{
    
    unsigned num_elements = vec.size();
    if (num_elements == 0)
        return NAN;
    
    float cur_max = vec[0];
    for (unsigned ix = 1; ix < num_elements; ix++)
    {
        float fetch = vec[ix];
        if (cur_max < fetch)
            cur_max = fetch;
    }
    
    return cur_max;
    
}

/// \brief Check if two floating point values are equal, within a certain
/// absolute tolerance
bool soft_eq(float x, float y, float tolerance)
{
    
    float diff = y - x;
    if (diff < 0.0f)
        diff = -diff;
    
    return diff < tolerance;
    
}

