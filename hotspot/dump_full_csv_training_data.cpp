/// \file
/// File Name:                      dump_full_csv_training_data.cpp \n
/// Date created:                   Fri Feb 17 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */

#include <fstream>
#include <iostream>
#include <string>

// See dump_training_data.h for documentation
unsigned dump_full_csv_training_data
(
    const float* temperature_matrix,
    unsigned width,
    unsigned height,
    const std::string& filename
){
    
    // Open file for training data
    std::ofstream file(filename.c_str());
    if (!file.good())
    {
        std::cerr
            << "Error.  Failed to open \""
            << filename
            << "\" for writing.\n";
        return 0;
    }
    
    unsigned linear_size = width * height;
    file << linear_size << '\n';
    for (unsigned ix = 0; ix < linear_size; ix++)
    {
        file << temperature_matrix[ix] << '\n';
    }
    
    // Return the number of training elements written
    return linear_size;
    
}

