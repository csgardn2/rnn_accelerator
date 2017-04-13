/// \file
/// File Name:                      fourier.h \n
/// Date created:                   Wed April 12 2017 \n
/// Engineers:                      Conor Gardner \n
/// Special Compile Instructions:   --std=c++11 \n
/// Compiler:                       g++ \n
/// Target OS:                      Ubuntu Linux 16.04 \n
/// Target architecture:            x86 64-bit \n */
/// See html/index.html for documentation.

#ifndef HEADER_GUARD_FOURIER
#define HEADER_GUARD_FOURIER

#include <cmath>
#include <vector>

/// \brief Compute the cosine coefficients a(t) for the given samples of
/// a given function.
/// \return The cosine coeficient a(frequency) for the given samples
float fourier_cos
(
    /// [in] Evaluate a function over one period at intervals of dt.
    const std::vector<float>& samples,
    
    /// [in] The frequency component which you want to find the
    /// amplitude.
    float frequency
);

#endif // header guard

