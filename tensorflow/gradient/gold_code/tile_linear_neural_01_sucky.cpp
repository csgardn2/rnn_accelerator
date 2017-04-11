/// \file
/// File Name:                      tile_linear_neural_01_sucky.cpp \n
/// Date created:                   Tues Dec 6 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n

#include <cstdint>

/// Neurally computet gradient method
uint8_t tile_linear_neural_01_sucky(const uint8_t in[9])
{

    
    // Training data = noise_512_input.png
    // Learning rate = 1e-2
    // Batches = 65536
    // Batch size = 512
    static const float trained_bias = 127.97548676;
    static const float trained_weights[9] = 
    {
        1.17935968e-04,
        4.97652143e-01,
        8.69434909e-04,
        -4.96767372e-01,
        -2.44064722e-04,
        4.97467011e-01,
        -4.91082028e-05,
        -4.97154921e-01,
        5.12720668e-04
    };
    
    signed ret = signed
    (
        trained_weights[0] * float(in[0])
      + trained_weights[1] * float(in[1])
      + trained_weights[2] * float(in[2])
      + trained_weights[3] * float(in[3])
      + trained_weights[4] * float(in[4])
      + trained_weights[5] * float(in[5])
      + trained_weights[6] * float(in[6])
      + trained_weights[7] * float(in[7])
      + trained_weights[8] * float(in[8])
      + trained_bias
    );
    
    if (ret < 0)
        ret = 0;
    if (ret > 255)
        ret = 255;
    
    return uint8_t(ret);
    
}

