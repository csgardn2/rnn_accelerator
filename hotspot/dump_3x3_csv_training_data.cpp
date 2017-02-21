/// \file
/// File Name:                      dump_3x3_csv_training_data.cpp \n
/// Date created:                   Wed Jan 18 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */

#include <fstream>
#include <iostream>
#include <string>

// See dump_training_data.h for documentation
unsigned dump_3x3_csv_training_data
(
    const float* source_temperature_matrix,
    const float* destination_temperature_matrix,
    unsigned width,
    unsigned height,
    const std::string& source_filename,
    const std::string& destination_filename
){
    
    // Open file for training inputs
    std::ofstream source_file(source_filename.c_str());
    if (!source_file.good())
    {
        std::cerr
            << "Error.  Failed to open \""
            << source_filename
            << "\" for writing.\n";
        return 0;
    }
    
    // Open file for training outputs
    std::ofstream destination_file(destination_filename.c_str());
    if (!destination_file.good())
    {
        std::cerr
            << "Error.  Failed to open \""
            << destination_filename
            << "\" for writing.\n";
        return 0;
    }
    
    unsigned linear_size = width * height;
    unsigned bound_x = width - 1;
    unsigned bound_y = height - 1;
    
    destination_file << linear_size << '\n';
    source_file << linear_size << '\n';
    
    for (unsigned iy = 0; iy < height; iy++)
    {
        
        // We will fetch a square 3x3 tile around the point (ix, iy).  Compute
        // pointers to each of the 3 rows in this tile.
        unsigned offset = iy * width;
        
        const float* destination_row = destination_temperature_matrix + offset;
        const float* source_row_locus = source_temperature_matrix + offset;
        // Don't go outside input.  Clamp to top row
        const float* source_row_above
            = (iy == 0)
            ? source_temperature_matrix
            : source_temperature_matrix + (offset - width);
        // Don't go outside input.  Clamp to bottom row
        const float* source_row_below
            = (iy == bound_y)
            ? source_temperature_matrix + offset
            : source_temperature_matrix + (offset + width);
        
        for (unsigned ix = 0; ix < width; ix++)
        {
            
            // Clamp left and right tile edges to the boundaries of the input
            // grid
            unsigned left = (ix == 0) ? 0 : ix - 1;
            unsigned right = (ix == bound_x) ? bound_x : ix + 1;
            
            destination_file << destination_row[ix] << '\n';
            source_file
                << source_row_above[left] << " " << source_row_above[ix] << " " << source_row_above[right] << " "
                << source_row_locus[left] << " " << source_row_locus[ix] << " " << source_row_locus[right] << " "
                << source_row_below[left] << " " << source_row_below[ix] << " " << source_row_below[right] << '\n';
            
        }
        
    }
    
    // Return the number of training elements written
    return linear_size;
    
}

