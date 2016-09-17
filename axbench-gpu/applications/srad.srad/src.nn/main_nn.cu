#include "../../../headers/activationFunction.h"

//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#include "define.c"
#include "extract_kernel.cu"
#include "prepare_kernel.cu"
#include "reduce_kernel.cu"
//#include "srad_kernel.cu"
#include "srad2_kernel.cu"
#include "compress_kernel.cu"
#include "graphics.c"
#include "resize.c"
#include "timer.c"

#include "device.c"				// (in library path specified to compiler)	needed by for device functions



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


    float parrotInput[5];
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

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80

	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;
	long long time11;
	long long time12;

	time0 = get_time();

    // inputs image, input paramenters
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements

    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;

    // counters
    int iter;   // primary loop
    long i,j;    // image row/col

	// memory sizes
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	//================================================================================80
	// 	GPU VARIABLES
	//================================================================================80

	// CUDA kernel execution parameters
	dim3 threads;
	int blocks_x;
	dim3 blocks;
	dim3 blocks2;
	dim3 blocks3;

	// memory sizes
	int mem_size;															// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* d_sums;															// partial sum
	fp* d_sums2;
	int* d_iN;
	int* d_iS;
	int* d_jE;
	int* d_jW;
	fp* d_dN;
	fp* d_dS;
	fp* d_dW;
	fp* d_dE;
	fp* d_I;																// input IMAGE on DEVICE
	fp* d_c;

	time1 = get_time();



#pragma parrot.start("srad")
	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 7){
		printf("ERROR: wrong number of arguments\n");
		return 0;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
	}

	time2 = get_time();

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80

    // read image
	image_ori_rows = Nr; // Amir
	image_ori_cols = Nc; // Amir
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	// read_graphics(	"../../../data/srad/image.pgm",
	// 							image_ori,
	// 							image_ori_rows,
	// 							image_ori_cols,
	// 							1);

	read_graphics( argv[5],
								image_ori,
								image_ori_rows,
								image_ori_cols,
								1);

	time3 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	Ne = Nr*Nc;

	image = (fp*)malloc(sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	time4 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

    r1     = 0;											// top row index of ROI
    r2     = Nr - 1;									// bottom row index of ROI
    c1     = 0;											// left column index of ROI
    c2     = Nc - 1;									// right column index of ROI

	// ROI image size
	NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size_i = sizeof(int) * Nr;											//
	iN = (int *)malloc(mem_size_i) ;										// north surrounding element
	iS = (int *)malloc(mem_size_i) ;										// south surrounding element
	mem_size_j = sizeof(int) * Nc;											//
	jW = (int *)malloc(mem_size_j) ;										// west surrounding element
	jE = (int *)malloc(mem_size_j) ;										// east surrounding element

	// N/S/W/E indices of surrounding pixels (every element of IMAGE)
	for (i=0; i<Nr; i++) {
		iN[i] = i-1;														// holds index of IMAGE row above
		iS[i] = i+1;														// holds index of IMAGE row below
	}
	for (j=0; j<Nc; j++) {
		jW[j] = j-1;														// holds index of IMAGE column on the left
		jE[j] = j+1;														// holds index of IMAGE column on the right
	}

	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
	iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
	iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1
	jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
	jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	//================================================================================80
	// 	GPU SETUP
	//================================================================================80

	// allocate memory for entire IMAGE on DEVICE
	mem_size = sizeof(fp) * Ne;																		// get the size of float representation of input IMAGE
	cudaMalloc((void **)&d_I, mem_size);														//

	// allocate memory for coordinates on DEVICE
	cudaMalloc((void **)&d_iN, mem_size_i);													//
	cudaMemcpy(d_iN, iN, mem_size_i, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **)&d_iS, mem_size_i);													//
	cudaMemcpy(d_iS, iS, mem_size_i, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **)&d_jE, mem_size_j);													//
	cudaMemcpy(d_jE, jE, mem_size_j, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **)&d_jW, mem_size_j);													//
	cudaMemcpy(d_jW, jW, mem_size_j, cudaMemcpyHostToDevice);			//

	// allocate memory for partial sums on DEVICE
	cudaMalloc((void **)&d_sums, mem_size);													//
	cudaMalloc((void **)&d_sums2, mem_size);												//

	// allocate memory for derivatives
	cudaMalloc((void **)&d_dN, mem_size);														//
	cudaMalloc((void **)&d_dS, mem_size);														//
	cudaMalloc((void **)&d_dW, mem_size);													//
	cudaMalloc((void **)&d_dE, mem_size);														//

	// allocate memory for coefficient on DEVICE
	cudaMalloc((void **)&d_c, mem_size);														//

	checkCUDAError("setup");

	//================================================================================80
	// 	KERNEL EXECUTION PARAMETERS
	//================================================================================80

	// all kernels operating on entire matrix
	threads.x = NUMBER_THREADS;												// define the number of threads in the block
	threads.y = 1;
	blocks_x = Ne/threads.x;
	if (Ne % threads.x != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;
	}
	blocks.x = blocks_x;													// define the number of blocks in the grid
	blocks.y = 1;

	time5 = get_time();

	//================================================================================80
	// 	COPY INPUT TO CPU
	//================================================================================80

	cudaMemcpy(d_I, image, mem_size, cudaMemcpyHostToDevice);

	time6 = get_time();

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

	extract<<<blocks, threads>>>(	Ne,
									d_I);

	checkCUDAError("extract");

	time7 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	// printf("iterations: ");

	// execute main loop
	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

	// printf("%d ", iter);
	// fflush(NULL);

		// execute square kernel
		prepare<<<blocks, threads>>>(	Ne,
										d_I,
										d_sums,
										d_sums2);

		checkCUDAError("prepare");

		// performs subsequent reductions of sums
		blocks2.x = blocks.x;												// original number of blocks
		blocks2.y = blocks.y;
		no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier

		while(blocks2.x != 0){

			checkCUDAError("before reduce");

			// run kernel
			reduce<<<blocks2, threads>>>(	Ne,
											no,
											mul,
											d_sums,
											d_sums2);

			checkCUDAError("reduce");

			// update execution parameters
			no = blocks2.x;												// get current number of elements
			if(blocks2.x == 1){
				blocks2.x = 0;
			}
			else{
				mul = mul * NUMBER_THREADS;									// update the increment
				blocks_x = blocks2.x/threads.x;								// number of blocks
				if (blocks2.x % threads.x != 0){							// compensate for division remainder above by adding one grid
					blocks_x = blocks_x + 1;
				}
				blocks2.x = blocks_x;
				blocks2.y = 1;
			}

			checkCUDAError("after reduce");

		}

		checkCUDAError("before copy sum");

		// copy total sums to device
		mem_size_single = sizeof(fp) * 1;
		cudaMemcpy(&total, d_sums, mem_size_single, cudaMemcpyDeviceToHost);
		cudaMemcpy(&total2, d_sums2, mem_size_single, cudaMemcpyDeviceToHost);

		checkCUDAError("copy sum");

		// calculate statistics
		meanROI	= total / fp(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / fp(NeROI)) - meanROI2;						// gets variance of ROI
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		// execute srad kernel
		srad<<<blocks, threads>>>(	lambda,									// SRAD coefficient
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									d_iN,									// indices of North surrounding pixels
									d_iS,									// indices of South surrounding pixels
									d_jE,									// indices of East surrounding pixels
									d_jW,									// indices of West surrounding pixels
									d_dN,									// North derivative
									d_dS,									// South derivative
									d_dW,									// West derivative
									d_dE,									// East derivative
									q0sqr,									// standard deviation of ROI
									d_c,									// diffusion coefficient
									d_I);									// output image

		checkCUDAError("srad");

		// execute srad2 kernel
		srad2<<<blocks, threads>>>(	lambda,									// SRAD coefficient
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									d_iN,									// indices of North surrounding pixels
									d_iS,									// indices of South surrounding pixels
									d_jE,									// indices of East surrounding pixels
									d_jW,									// indices of West surrounding pixels
									d_dN,									// North derivative
									d_dS,									// South derivative
									d_dW,									// West derivative
									d_dE,									// East derivative
									d_c,									// diffusion coefficient
									d_I);									// output image

		checkCUDAError("srad2");

	}

	// printf("\n");

	time8 = get_time();

	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

	compress<<<blocks, threads>>>(	Ne,
									d_I);

	checkCUDAError("compress");

	time9 = get_time();

	//================================================================================80
	// 	COPY RESULTS BACK TO CPU
	//================================================================================80

	cudaMemcpy(image, d_I, mem_size, cudaMemcpyDeviceToHost);

	checkCUDAError("copy back");

	time10 = get_time();

	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	write_graphics(	argv[6],
					image,
					Nr,
					Nc,
					1,
					255);

	time11 = get_time();

	//================================================================================80
	//	DEALLOCATE
	//================================================================================80

	free(image_ori);
	free(image);
	free(iN);
	free(iS);
	free(jW);
	free(jE);

	cudaFree(d_I);
	cudaFree(d_c);
	cudaFree(d_iN);
	cudaFree(d_iS);
	cudaFree(d_jE);
	cudaFree(d_jW);
	cudaFree(d_dN);
	cudaFree(d_dS);
	cudaFree(d_dE);
	cudaFree(d_dW);
	cudaFree(d_sums);
	cudaFree(d_sums2);

	time12 = get_time();

	//================================================================================80
	//	DISPLAY TIMING
	//================================================================================80

	// printf("Time spent in different stages of the application:\n");
	// printf("%15.12f s, %15.12f % : SETUP VARIABLES\n", 														(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : READ COMMAND LINE PARAMETERS\n", 										(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : READ IMAGE FROM FILE\n", 												(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : RESIZE IMAGE\n", 														(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n", 					(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : COPY DATA TO CPU->GPU\n", 												(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : EXTRACT IMAGE\n", 														(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : COMPUTE\n", 																(float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : COMPRESS IMAGE\n", 														(float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : COPY DATA TO GPU->CPU\n", 												(float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : SAVE IMAGE INTO FILE\n", 												(float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
	// printf("%15.12f s, %15.12f % : FREE MEMORY\n", 															(float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
	// printf("Total time:\n");
	// printf("%.12f s\n", 																					(float) (time12-time0) / 1000000);
#pragma parrot.end("srad")
}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
