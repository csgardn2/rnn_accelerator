/// \file
/// File Name:                      main.cpp \n
/// Date created:                   Wed Jan 25 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.

#include <fstream>
#include <iostream>
#include <random>

/// Usage:
///     ./training_data_generator [iterations] [training_output_file.py]
int main(int argc, char** argv)
{
    
    // Parse command line arguments
    if (argc != 3)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " [iterations] [output_file.py]\n";
        return -1;
    }
    unsigned iterations = atoi(argv[1]);
    if (iterations < 2)
    {
        std::cerr << "Error.  Iteration count must be at least 2\n";
        return -1;
    }
    std::ofstream file(argv[2]);
    if (!file.good())
    {
        std::cerr
            << "Error.  Failed to open \""
            << argv[3]
            << "\" for writing.\n";
        return -1;
    }
    
    // Generate input-output training sequence in RAM
    // so that both the inputs and outputs can be dumped to the same file
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned> distribution(0, 255);
    float* input_sequence = new float[iterations];
    float* output_sequence = new float[iterations];
    
    input_sequence[0] = 0;
    output_sequence[0] = 0.0f;
    float state = 0.0f;
    for (unsigned ix = 1; ix < iterations; ix++)
    {
        
        float cur_input = float(distribution(generator));
        input_sequence[ix] = cur_input;
        
        state = 0.5 * state + cur_input;
        output_sequence[ix] = state;
        
    }
    
    // Dump the input and output sequences as python arrays
    // to be used as training data for tensorflow
    file
        << "#!/usr/bin/python"
        << "input_sequence_gold_size = "
        << iterations
        << "\noutput_sequence_gold_size = "
        << iterations
        << "\ninput_sequence_gold = [";
    unsigned last_ix = iterations - 1;
    for (unsigned ix = 0; ix < last_ix; ix++)
        file << input_sequence[ix] << ", ";
    file << input_sequence[last_ix] << "]\noutput_sequence_gold = [";
    for (unsigned ix = 0; ix < last_ix; ix++)
        file << output_sequence[ix] << ", ";
    file << output_sequence[last_ix] << "]\n";
    
    delete[] input_sequence;
    delete[] output_sequence;
    
    return 0;
    
}

