/// \file
/// File Name:                      tile_nonlinear_neural_01_sucky.cpp \n
/// Date created:                   Tues Dec 6 2016 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n

#include <cstdint>

/// Neurally computet gradient method
uint8_t tile_nonlinear_neural_01_sucky(const uint8_t in[9])
{
    
    // From an external neural network trainer
    static const float trained_weights[9] = 
    {
        -0.05729901,
        -0.01248036,
         0.00629563,
        -0.08958419,
        -0.00523720,
         0.12242614,
        -0.04516511,
         0.05326009,
         0.06032013
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
    );
    
    if (ret < 0)
        ret = 0;
    if (ret > 255)
        ret = 255;
    
    return uint8_t(ret);
    
}

