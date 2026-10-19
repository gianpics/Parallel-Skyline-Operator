
/****************************************************************************
 *
 * cuda-skyline.cu - CUDA implementaiton of the skyline operator
 *
 * Per compilare:
 *
 *      nvcc -Wno-deprecated-gpu-targets cuda-skyline.cu -o cuda-skyline -lm
 *
 * Per eseguire il programma:
 *
 *      ./cuda-skyline < input > output
 *
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "hpc.h"

typedef struct {
    float *P;   /* coordinates P[i][j] of point i               */
    int N;      /* Number of points (rows of matrix P)          */
    int D;      /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input( points_t *points )
{
    char buf[1024];
    int N, D;
    float *P;

    if (1 != scanf("%d", &D)) {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin)) { /* ignore rest of the line */
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N)) {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float*)malloc( D * N * sizeof(*P) );
    assert(P);
    for (int i=0; i<N; i++) {
        for (int k=0; k<D; k++) {
            if (1 != scanf("%f", &(P[i*D + k]))) {
                fprintf(stderr, "FATAL: failed to get coordinate %d of point %d\n", k, i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points( points_t* points )
{
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

/* Returns 1 iff |p| dominates |q| */
__device__ int dominates( const float * p, const float * q, int D )
{
    /* The following loops could be merged, but the keep them separated
       for the sake of readability */
    for (int k=0; k<D; k++) {
        if (p[k] < q[k]) {
            return 0;
        }
    }
    for (int k=0; k<D; k++) {
        if (p[k] > q[k]) {
            return 1;
        }
    }
    return 0;
}

__global__ void init_s(int *s, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) 
    {
        s[idx] = 1;
    }
}


__global__ void skyline(const float* P, int* S, const int N, const int D){
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(idx < N) {
        bool is_skyline = true;
        for(int j=0; j<N; j++){
            if ( S[j] && dominates( &(P[j*D]), &(P[idx*D]), D ) ) {
                is_skyline = false;
            }
        }

        if(!is_skyline)
            S[idx] = 0;
    }
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard output. `s[i] == 1` iff point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline( const points_t* points, const int *s, int r )
{
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i=0; i<N; i++) {
        if ( s[i] ) {
            for (int k=0; k<D; k++) {
                printf("%f ", P[i*D + k]);
            }
            printf("\n");
        }
    }
}

int main( int argc, char* argv[] )
{
    points_t points;

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);
    size_t SIZE_INT   = (size_t)points.N * sizeof(int);
    size_t SIZE_FLOAT = (size_t)points.N * (size_t)points.D * sizeof(float);
    int *s = (int*)malloc(SIZE_INT);
    assert(s);

    /* device copies */
    float *d_p;
    int *d_s;

    /* allocating device copies */
    cudaSafeCall( cudaMalloc((void **)&d_p, SIZE_FLOAT) );
    cudaSafeCall( cudaMalloc((void **)&d_s, SIZE_INT) );

    
    const int BLKDIM = 1024;

    const int NBLOCKS = (points.N + BLKDIM - 1) / BLKDIM;

    const double tstart = hpc_gettime();

    // init s
    init_s<<<NBLOCKS, BLKDIM>>>(d_s, points.N);

    /* copying to device*/
    cudaSafeCall( cudaMemcpy(d_p, points.P, SIZE_FLOAT, cudaMemcpyHostToDevice) );

    skyline<<<NBLOCKS, BLKDIM>>>(d_p, d_s, points.N, points.D);

    cudaSafeCall( cudaMemcpy(s, d_s, SIZE_INT, cudaMemcpyDeviceToHost) );

    /* count r  */
    int r = 0;
    for(int i=0; i < points.N; i++){
        if(s[i] == 1)
            r++;
    }

    cudaDeviceSynchronize();

    const double elapsed = hpc_gettime() - tstart;
    print_skyline(&points, s, r);

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n\n", r);
    fprintf(stderr, "Execution time (s) %f\n", elapsed);

    free_points(&points);
    free(s);
    return EXIT_SUCCESS;
}
