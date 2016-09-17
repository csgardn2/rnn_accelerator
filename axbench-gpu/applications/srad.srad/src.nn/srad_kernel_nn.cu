#include "../../../headers/activationFunction.h"

// BUG IN SRAD APPLICATIONS SEEMS TO BE SOMEWHERE IN THIS CODE, WRONG MEMORY ACCESS

// srad kernel
__global__ void srad(	fp d_lambda,
									 int d_Nr,
									 int d_Nc,
									 long d_Ne,
									 int *d_iN,
									 int *d_iS,
									 int *d_jE,
									 int *d_jW,
									 fp *d_dN,
									 fp *d_dS,
									 fp *d_dE,
									 fp *d_dW,
									 fp d_q0sqr,
									 fp *d_c,
									 fp *d_I){

	// indexes
    int bx = blockIdx.x;													// get current horizontal block index
	int tx = threadIdx.x;													// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;													// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;												// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															// make sure that only threads matching jobs run

		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element

		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative


    float parrotInput[9];
    float parrotOutput[1];

    parrotInput[0] = d_Jc;
    parrotInput[1] = d_dN_loc;
    parrotInput[2] = d_dS_loc;
    parrotInput[3] = d_dW_loc;
    parrotInput[4] = d_dE_loc;

float layer_1_0 = parrotInput[0] * 1.234254 + parrotInput[1] * -4.005687 + parrotInput[2] * 2.788479 + parrotInput[3] * 2.507076 + parrotInput[4] * -12.129114 + 1.0f * 2.448057;

float layer_1_1 = parrotInput[0] * 1.007452 + parrotInput[1] * -1.526600 + parrotInput[2] * 6.647075 + parrotInput[3] * -0.814735 + parrotInput[4] * 5.108116 + 1.0f * 2.181065;

float layer_1_2 = parrotInput[0] * 1.014252 + parrotInput[1] * 6.239608 + parrotInput[2] * -1.567504 + parrotInput[3] * 2.511521 + parrotInput[4] * -0.184482 + 1.0f * 1.828247;

float layer_1_3 = parrotInput[0] * -0.290943 + parrotInput[1] * -0.596419 + parrotInput[2] * -0.717598 + parrotInput[3] * -0.373179 + parrotInput[4] * 0.177434 + 1.0f * 0.148764;

float layer_2_0 = sigmoid(layer_1_0, 0.500000) * 1.020804 + sigmoid(layer_1_1, 0.500000) * 1.004395 + sigmoid(layer_1_2, 0.500000) * 1.056305 + sigmoid(layer_1_3, 0.500000) * 0.908768 + 1.0f * 0.955657;

layer_2_0 = linear(layer_2_0, 0.5);

parrotOutput[0] = layer_2_0;

// parrotOutput[0] = layer_2_0;
// 
// 		// normalized discrete gradient mag squared (equ 52,53)
// 		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)
// 
// 		// normalized discrete laplacian (equ 54)
// 		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)
// 
// 		// ICOV (equ 31/35)
// 		d_num  = (0.5*d_G2) - ((1.0/16.0)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
// 		d_den  = 1 + (0.25*d_L);												// den (based on laplacian)
// 		d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)
// 
// 		// diffusion coefficent (equ 33) (every element of IMAGE)
// 		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
// 		d_c_loc = 1.0 / (1.0+d_den) ;										// diffusion coefficient (based on den)
// 
// 		parrotOutput[0] = d_c_loc;
// 
// 
// #pragma parrot(output, "srad", [1]<-2.0; 2.0>parrotOutput)

		d_c_loc = parrotOutput[0];

		// saturate diffusion coefficent to 0-1 range
		if (d_c_loc < 0){													// if diffusion coefficient < 0
			d_c_loc = 0;													// ... set to 0
		}
		else if (d_c_loc > 1){												// if diffusion coefficient > 1
			d_c_loc = 1;													// ... set to 1
		}

		// save data to global memory
		d_dN[ei] = d_dN_loc;
		d_dS[ei] = d_dS_loc;
		d_dW[ei] = d_dW_loc;
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;

	}

}
