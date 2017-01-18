/// \file
/// File Name:                      utility.h \n
/// Date created:                   Wed Jan 18 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or utility.cpp for documentation.
 
#ifndef HEADER_GUARD_UTILITY
#define HEADER_GUARD_UTILITY

#include <vector>

/// \brief Used to return values from \ref pick_color
class color_t
{
    
    public:
        
        /// \brief Default constructor
        color_t() = default;
        
        /// \brief Initialization constructor
        color_t(unsigned char red_, unsigned char green_, unsigned char blue_)
            :   red(red_), green(green_), blue(blue_)
        {
            // Intentionally left blank
        }
        
        /// \brief Red channel of a pixel
        unsigned char red;
        
        /// \brief Green channel of a pixel
        unsigned char green;
        
        /// \brief Blue channel of a pixel
        unsigned char blue;
        
};

float triangle(float x, float min, float max);
color_t pick_color(float temperature, float min_temperature, float max_temperature);
float min(const std::vector<float>& vec);
float max(const std::vector<float>& vec);
bool soft_eq(float x, float y, float tolerance = 0.0001);

#endif // header guard

