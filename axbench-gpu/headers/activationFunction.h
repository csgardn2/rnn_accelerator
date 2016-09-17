// Amir Yazdanbakhsh
// January 31st, 2015
// a.yazdanbakhsh@gatech.edu

#ifndef __ACTIVATION_FUNCTION_H_
#define __ACTIVATION_FUNCTION_H_

// Linear Activation Function
__device__ float linear(float x, float s)
{
	return ( x * s);
}

__device__ float sigmoid(float x, float s)
{
	return ( 1.0 / ( 1 + expf(-2.0 * x * s)));
}

__device__ float symmetricSigmoid(float x, float s)
{
	return ( ( 2.0 / ( 1 + expf(-2.0 * x * s))) -  1.0 );
}

#endif