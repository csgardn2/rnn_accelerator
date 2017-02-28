/// \file
/// File Name:                      color_picker.cpp \n
/// Date created:                   Wed Jan 18 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or for documentation.

#include "color_picker.h"

/// \brief return the y-value of a point on an isosceles triangle of height 1
/// with the other two points on the x-axis with x-value min and max.
float triangle(float x, float min, float max)
{
    
    float normalized = (x - min) / (max - min);    
    
    if (normalized < 0.0f || normalized > 1.0f)
        return 0.0f;
    else if (normalized < 0.5f)
        return 2.0f * normalized;
    else
        return 2.0f * (1.0f - normalized);
    
}

/// \brief Do a 5-point linear interpolation from black to blue to green to red
/// to white with black being the coldest (min_temperature) and white being
/// the hottest (max_temperature)
color_t pick_color(float temperature, float min_temperature, float max_temperature)
{
    
    if (temperature < min_temperature)
        return color_t(0, 0, 0);
    if (temperature > max_temperature)
        return color_t(255, 255, 255);
    
    float range = max_temperature - min_temperature;
    float interpolation_endpoint_0 = min_temperature;
    float interpolation_endpoint_1 = min_temperature + range * 0.1667;
    float interpolation_endpoint_2 = min_temperature + range * 0.3333;
    float interpolation_endpoint_3 = min_temperature + range * 0.5;
    float interpolation_endpoint_4 = min_temperature + range * 0.6667;
    float interpolation_endpoint_5 = min_temperature + range * 0.8333;
    float interpolation_endpoint_6 = max_temperature;
    float interpolation_endpoint_7 = min_temperature + range * 1.1667;
    
    float blue   = triangle(temperature, interpolation_endpoint_0, interpolation_endpoint_2);
    float cyan   = triangle(temperature, interpolation_endpoint_1, interpolation_endpoint_3);
    float green  = triangle(temperature, interpolation_endpoint_2, interpolation_endpoint_4);
    float yellow = triangle(temperature, interpolation_endpoint_3, interpolation_endpoint_5);
    float red    = triangle(temperature, interpolation_endpoint_4, interpolation_endpoint_6);
    float white  = triangle(temperature, interpolation_endpoint_5, interpolation_endpoint_7);
    
    signed cast_red = signed((yellow + red + white) * 256.0f);
    signed cast_green = signed((cyan + green + yellow + white) * 256.0f);
    signed cast_blue = signed((blue + cyan + white) * 256.0f);
    
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

