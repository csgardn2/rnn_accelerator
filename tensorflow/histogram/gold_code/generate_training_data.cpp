/// \file
/// File Name:                      generate_trainig_data.cpp \n
/// Date created:                   Mon Dec 12 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.

#include <cstdlib>
#include <iostream>
#include <fstream>

/// Usage:
///     ./generate_training_data
///         [number of training sample]
///         [number of input elements per sample]
///         [number of histogram output bins per sample]
///         [filename to write input samples to]
///         [filename to write output histograms to]
int main(int argc, char** argv)
{
    
    // Argument number check
    if (argc != 6)
    {
        std::cerr
            << "Error.  You must specify exactly 5 command line arguments\n"
               "    [number of training samples to generate]\n"
               "    [number of input elements per sample]\n"
               "    [number of histogram output bins per sample]\n"
               "    [filename to write input samples to]\n"
               "    [filename to write output histograms to]\n";
        return -1;
    }
    
    // Argument 1 (leftmost)
    signed num_training_samples = atoi(argv[1]);
    if (num_training_samples < 1)
    {
        std::cerr
            << "Error.  You must generate at least one training sample."
               "You requested "
            << argv[1]
            << " samples\n";
        return -1;
    }
    
    // Argument 2
    signed inputs_per_sample = atoi(argv[2]);
    if (inputs_per_sample < 1)
    {
        std::cerr
            << "Error.  You must generate at least one input per sample."
               "You requested "
            << argv[1]
            << '\n';
        return -1;
    }
    
    // Argument 3
    signed num_histogram_bins = atoi(argv[3]);
    if (num_histogram_bins < 1)
    {
        std::cerr
            << "Error.  You must generate at least one output per sample."
               "You requested "
            << argv[1]
            << '\n';
        return -1;
    }
    
    // Argument 4
    std::ofstream input_samples_file(argv[4]);
    if (!input_samples_file.good())
    {
        std::cerr << "Error.  Failed to open \"" << argv[4] << "\" for writing\n";
        return -1;
    }
    
    // Argument 5
    std::ofstream histograms_file(argv[5]);
    if (!histograms_file.good())
    {
        std::cerr << "Error.  Failed to open \"" << argv[5] << "\" for writing\n";
        return -1;
    }
    
    srand(42);
    
    // Write out the beginning of the output files (start python arrays)
    input_samples_file
        << "# number or samples = " << num_training_samples << '\n'
        << "# inputs per sample = " << inputs_per_sample << '\n'
        << "# number of histogram bins = " << num_histogram_bins << '\n'
        << "# histogram bin size = 1\n"
           "input_samples = [\n";
    
    histograms_file
        << "# number or samples = " << num_training_samples << '\n'
        << "# inputs per sample = " << inputs_per_sample << '\n'
        << "# number of histogram bins = " << num_histogram_bins << '\n'
        << "# histogram bin size = 1\n"
           "output_histograms = [\n";
    
    // Generate a bunch of random data points and create their histograms
    unsigned* cur_input_sample = new unsigned[inputs_per_sample];
    unsigned* cur_histogram = new unsigned[num_histogram_bins];
    for (signed iy = 0; iy < num_training_samples; iy++)
    {
        // Generate a random input sample
        for (signed ix = 0; ix < inputs_per_sample; ix++)
            cur_input_sample[ix] = rand() % num_histogram_bins;
        
        // Generate a histogram (the output sample)
        for (signed ix = 0; ix < num_histogram_bins; ix++)
            cur_histogram[ix] = 0;
        for (signed ix = 0; ix < inputs_per_sample; ix++)
            cur_histogram[cur_input_sample[ix]]++;
        
        // Too lazy to hoist conditionals out of the loop
        bool enable_comma = iy != (num_training_samples - 1);
        
        // Dump input samples file
        input_samples_file << '[';
        signed input_bound = inputs_per_sample - 1;
        for (signed ix = 0; ix < input_bound; ix++)
            input_samples_file << cur_input_sample[ix] << ", ";
        input_samples_file << cur_input_sample[input_bound] << ']';
        if (enable_comma)
            input_samples_file << ',';
        input_samples_file << '\n';
        
        // Dump histogram file
        histograms_file << '[';
        signed histogram_bound = num_histogram_bins - 1;
        for (signed ix = 0; ix < histogram_bound; ix++)
            histograms_file << cur_histogram[ix] << ", ";
        histograms_file << cur_histogram[histogram_bound] << ']';
        if (enable_comma)
            histograms_file << ',';
        histograms_file << '\n';
        
    }
    
    // Finish output files
    input_samples_file << "]\n\n";
    histograms_file << "]\n\n";
    
    delete[] cur_histogram;
    delete[] cur_input_sample;
    
    // Output files flushed to disk and cleaned up
    return 0;
    
}

