#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#ifdef RD_WG_SIZE_0_0                                                            
    #define BLOCK_SIZE RD_WG_SIZE_0_0                                        
#elif defined(RD_WG_SIZE_0)                                                      
    #define BLOCK_SIZE RD_WG_SIZE_0                                          
#elif defined(RD_WG_SIZE)                                                        
    #define BLOCK_SIZE RD_WG_SIZE                                            
#else                                                                                    
    #define BLOCK_SIZE 16                                                            
#endif                                                                                   

#define STR_SIZE 256

/* Maximum power density possible (say 300W for a 10mm x 10mm chip) */
#define MAX_PD  (3.0e6)

/* Required precision in degrees    */
#define PRECISION   0.001

#define SPEC_HEAT_SI 1.75e6
#define K_SI 100

// Capacitance fitting factor
#define FACTOR_CHIP 0.5

// Add one iteration will extend the pyramid base by 2 per each borderline
# define EXPAND_RATE 2

/* chip parameters  */
const float t_chip = 0.0005;
const float chip_height = 0.016;
const float chip_width = 0.016;
/* Ambient temperature, assuming no package at all  */
float amb_temp = 80.0;

void run(int argc, char** argv);

/* Define timer macros */
#define pin_stats_reset()         startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)

void 
fatal(const char *s)
{
    fprintf(stderr, "error: %s\n", s);
}

void writeoutput(float *vect, int grid_rows, int grid_cols, const char *file)
{
    
    int i, j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if((fp = fopen(file, "w" )) == 0)
        printf("The file was not opened\n");


    for (i = 0; i < grid_rows; i++)
    {
        for (j = 0; j < grid_cols; j++)
        {

         sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
         fputs(str,fp);
         index++;
        }
    }
    
    fclose(fp);
    
}

void readinput(float *vect, int grid_rows, int grid_cols, const char *file)
{


    FILE *fp = fopen(file, "r" );
    if (fp == NULL)
        printf("The file was not opened\n");
    
    char str[STR_SIZE];
    for (int iy = 0; iy < grid_rows; iy++) 
    {
        float* row = vect + iy * grid_cols;
        for (int ix = 0; ix < grid_cols; ix++)
        {
            fgets(str, STR_SIZE, fp);
            if (feof(fp))
                fatal("not enough lines in file");
            float val;
            if ((sscanf(str, "%f", &val) != 1))
                fatal("invalid file format");
            row[ix] = val;
        }
    }
    
    fclose(fp); 

}

#define IN_RANGE(x, min, max)   ((x) >= (min) && (x) <= (max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x > (max)) ? max : x )
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void calculate_temp
(
    int iteration,      // number of iterations
    float *power,       // power input
    float *temp_src,    // temperature input/output
    float *temp_dst,    // temperature input/output
    int grid_cols,      // Number of grid tiles in a single row (width)
    int grid_rows,      // Number of grid tiles in a single column (height)
    int border_cols,    // border offset 
    int border_rows,    // border offset
    float Cap,          // Heat capacitance of a single grid tile
    float Rx, 
    float Ry, 
    float Rz, 
    float step, 
    float time_elapsed
){
    
    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    float amb_temp = 80.0f;
    
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float step_div_Cap = step / Cap;
    
    float Rx_1 = 1.0f / Rx;
    float Ry_1 = 1.0f / Ry;
    float Rz_1 = 1.0f / Rz;
    
    // each block finally computes result for a small block
    // after N iterations. 
    // it is the non-overlapping small blocks that cover 
    // all the input data

    // calculate the small block size
    int small_block_rows = BLOCK_SIZE - iteration * 2;  //EXPAND_RATE
    int small_block_cols = BLOCK_SIZE - iteration * 2;  //EXPAND_RATE

    // calculate the boundary for the block according to 
    // the boundary of its small block
    int blkY = small_block_rows * by - border_rows;
    int blkX = small_block_cols * bx - border_cols;
    int blkYmax = blkY + BLOCK_SIZE - 1;
    int blkXmax = blkX + BLOCK_SIZE - 1;

    // calculate the global thread coordination
    int yidx = blkY + ty;
    int xidx = blkX + tx;

    // load data if it is within the valid input range
    int loadYidx = yidx;
    int loadXidx = xidx;
    int index = grid_cols * loadYidx + loadXidx;
    
    // Fetch a temperature and power tile from global memory to fast shared memory
    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols - 1))
    {
        temp_on_cuda[ty][tx] = temp_src[index];
        power_on_cuda[ty][tx] = power[index];
    }
    
    __syncthreads();

    // Effective range within this block that falls within 
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax-grid_rows + 1) : BLOCK_SIZE - 1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;
    
    int N = ty - 1;
    int S = ty + 1;
    int W = tx - 1;
    int E = tx + 1;
    
    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;
    
    bool computed;
    for (int i = 0; i < iteration; i++)
    {
         
        computed = false;
        if
        (
            IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2)
         && IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2)
         && IN_RANGE(tx, validXmin, validXmax)
         && IN_RANGE(ty, validYmin, validYmax)
        ){
            
            computed = true;
            temp_t[ty][tx] =
                temp_on_cuda[ty][tx]
              + step_div_Cap *
                (
                    power_on_cuda[ty][tx]
                  + (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0 * temp_on_cuda[ty][tx]) * Ry_1
                  + (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0 * temp_on_cuda[ty][tx]) * Rx_1
                  + (amb_temp - temp_on_cuda[ty][tx]) * Rz_1
                );
            
        }
        
        __syncthreads();
        
        if (i == iteration - 1)
            break;
        
        if (computed)     //Assign the computation range
            temp_on_cuda[ty][tx] = temp_t[ty][tx];
            
        __syncthreads();
        
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the 
    // small block perform the calculation and switch on ``computed''
    if (computed)
        temp_dst[index] = temp_t[ty][tx];      
     
}

// compute N time steps
int compute_tran_temp
(
    float *MatrixPower,
    float *MatrixTemp[2],
    int col,
    int row,
    int total_iterations,
    int num_iterations,
    int blockCols,
    int blockRows,
    int borderCols,
    int borderRows
){
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);  
    
    float grid_height = chip_height / row;
    float grid_width = chip_width / col;
    
    // Heat capacity of a single grid tile on the chip
    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    
    // Thermal resistance across a single grid tile along a given direction
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float time_elapsed = 0.001;;
    
    // Double buffering indexes
    int src = 1;
    int dst = 0;
    
    for (int t = 0; t < total_iterations; t += num_iterations)
    {
        int temp = src;
        src = dst;
        dst = temp;
        calculate_temp
            <<<dimGrid, dimBlock>>>
            (
                MIN(num_iterations, total_iterations - t),
                MatrixPower,
                MatrixTemp[src],
                MatrixTemp[dst],
                col,
                row,
                borderCols,
                borderRows,
                Cap,
                Rx,
                Ry,
                Rz,
                step,
                time_elapsed
            );
    }
    
    return dst;
    
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<pyramid_height> - pyramid height (positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char** argv)
{
    printf("WG size of kernel = %d X %d\n", BLOCK_SIZE, BLOCK_SIZE);
    run(argc,argv);
    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    
    if (argc != 7)
        usage(argc, argv);
    
    int grid_rows;
    int grid_cols;
    int total_iterations;
    int pyramid_height; // number of iterations
    if
    (
        (grid_rows = atoi(argv[1])) <= 0
    ||  (grid_cols = atoi(argv[1])) <= 0
    ||  (pyramid_height = atoi(argv[2])) <= 0
    ||  (total_iterations = atoi(argv[3])) <= 0
    ){
        usage(argc, argv);
    }
    
    const char* tfile = argv[4];
    const char* pfile = argv[5];
    const char* ofile = argv[6];
    
    int size = grid_rows * grid_cols;

    /* --------------- pyramid parameters --------------- */
    int borderCols = pyramid_height * EXPAND_RATE / 2;
    int borderRows = pyramid_height * EXPAND_RATE / 2;
    int smallBlockCol = BLOCK_SIZE - pyramid_height * EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE - pyramid_height * EXPAND_RATE;
    int blockCols = grid_cols / smallBlockCol + ((grid_cols % smallBlockCol == 0) ? 0 : 1);
    int blockRows = grid_rows / smallBlockRow + ((grid_rows % smallBlockRow == 0) ? 0 : 1);
    
    float* FilesavingTemp = (float*)malloc(size * sizeof(float));
    float* FilesavingPower = (float*)malloc(size * sizeof(float));
    float* MatrixOut = (float*)calloc(size, sizeof(float));

    if (FilesavingPower == NULL || FilesavingTemp == NULL || MatrixOut == NULL)
        fatal("unable to allocate memory");
    
    printf
    (
        "pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",
        pyramid_height,
        grid_cols,
        grid_rows,
        borderCols,
        borderRows,
        blockCols,
        blockRows,
        smallBlockCol,
        smallBlockRow
    );
    
    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);
    
    // Double buffering.
    // MatrixTemp[0] is initialized from file and used as an input during the
    // first iteration while the next temperature grid is written to
    // MatrixTemp[1].  After the first iteration is complete, the two pointers
    // are swapped and the previous output becomes the new input and the old
    // input is overwritten.
    float *MatrixTemp[2];
    cudaMalloc((void**)&MatrixTemp[0], sizeof(float) * size);
    cudaMalloc((void**)&MatrixTemp[1], sizeof(float) * size);
    cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float) * size, cudaMemcpyHostToDevice);
    
    float *MatrixPower;
    cudaMalloc((void**)&MatrixPower, sizeof(float) * size);
    cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float) * size, cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp
    (
        MatrixPower,
        MatrixTemp,
        grid_cols,
        grid_rows,
        total_iterations,
        pyramid_height,
        blockCols,
        blockRows,
        borderCols,
        borderRows
    );
    printf("Ending simulation\n");
    cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float) * size, cudaMemcpyDeviceToHost);

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    cudaFree(MatrixPower);
    cudaFree(MatrixTemp[0]);
    cudaFree(MatrixTemp[1]);
    free(MatrixOut);
    
}

