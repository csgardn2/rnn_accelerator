/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Wed April 12 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html or TODO.h for documentation.

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include "fourier.h"

float f(float x)
{
    return
        4.0f
      + 3.0f * std::cos(2 * M_PI * x)
      + 7.0f * std::cos(4 * M_PI * x);
}

/// Usage:
///     TODO - command line usage
int main()
{
    
    static const unsigned num_functions = 65536;
    static const unsigned samples_per_function = 256;
    static const float dt = 1.0f / float(samples_per_function);
    
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, 255);
    
    // Start creating new python files to be filled with training data
    std::ofstream training_input_file("input_functions.py");
    std::ofstream training_output_file("output_coefficients.py");
    
    training_input_file
     << "# 2D python array of training inputs"
     << "# number of functions = " << num_functions << '\n'
     << "# samples per functions = " << samples_per_function << '\n'
     << "# input_functions[number of functions][samples per function]\n"
     << "input_functions = [\n";
    
    training_output_file
     << "# 2D python array of training outputs"
     << "# number of functions = " << num_functions << '\n'
     << "# coefficients per function = 8\n"
     << "# input_functions[number of functions][8]\n"
     << "output_coefficients = [\n";
    
    // Write the first data points to the python files
    std::vector<float> samples(samples_per_function);
    for (unsigned iy = 0, last_function = num_functions - 1; iy < num_functions; iy++)
    {
        
        // Generate some random cosine coefficients
        float a0 = float(distribution(generator));
        float a1 = float(distribution(generator));
        float a2 = float(distribution(generator));
        float a3 = float(distribution(generator));
        float a4 = float(distribution(generator));
        float a5 = float(distribution(generator));
        float a6 = float(distribution(generator));
        float a7 = float(distribution(generator));
        
        training_output_file
         << '['
         << a0 << ", "
         << a1 << ", "
         << a2 << ", "
         << a3 << ", "
         << a4 << ", "
         << a5 << ", "
         << a6 << ", "
         << a7 << ']';
        
        if (iy == last_function)
            training_output_file << '\n';
        else
            training_output_file << ",\n";
        
        // Sample the random cosine sum function to generate training
        // inputs
        float t = 0.0f;
        training_input_file << '[';
        for (unsigned ix = 0, last = samples_per_function - 1; ix < samples_per_function; ix++)
        {
            
            float sample = 
                a0
              + a1 * std::cos(2 * M_PI * t)
              + a2 * std::cos(4 * M_PI * t)
              + a3 * std::cos(6 * M_PI * t)
              + a4 * std::cos(8 * M_PI * t)
              + a5 * std::cos(10 * M_PI * t)
              + a6 * std::cos(12 * M_PI * t)
              + a7 * std::cos(14 * M_PI * t);
            
            samples[ix] = sample;
            
            // Feel free to hoist these conditionals from the loop as
            // an optimization later
            if (ix == last)
                if (iy == last_function)
                    training_input_file << sample << "]\n";
                else
                    training_input_file << sample << "],\n";
            else
                training_input_file << sample << ", ";
            
            t += dt;
            
        }
         
    }
    
    training_input_file << "]\n";
    training_output_file << "]\n";
    
    return 0;
    
}
