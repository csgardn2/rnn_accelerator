#ifndef __PARROT_DATA_EXTERN_H_
#define __PARROT_DATA_EXTERN_H_

// Parrot Observer Data Storage
#ifndef PARROTSIZE
	#define PARROTSIZE 10000000
#endif
extern __device__ float dData[PARROTSIZE];
extern __device__ int dIndex;

#endif