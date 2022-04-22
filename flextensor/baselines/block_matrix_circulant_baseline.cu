#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_runtime.h>
#include <malloc.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <getopt.h>

#define fail(_code) do {\
    if ((_code) != cudaSuccess)\
        goto cleanup;\
} while(0)
#define panic(_code) assert((_code) == cudaSuccess)
#define checkpoint fprintf(stderr, "%s:%d: checkpoint\n", __FILE__, __LINE__);


#define MAX_NDEV 4
static cudaDeviceProp props[MAX_NDEV];
static int ndev = 0;
static int dev_used = 0;

static int number = 10;

// use just one dim
__global__ void 
BMC_kernel(const float *Input, float *Output, const int ROW, const int COL, const int FFT)
{
#define IDX(_i, _j) ((_i) * COL + (_j))

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    if (id >= (ROW / FFT) * FFT * (COL / FFT))
        return;

    int col = id % (COL / FFT * FFT);
    int col_start = (col / FFT) * FFT;
    int col_id = col % FFT;

    int row_start = (id / (COL / FFT * FFT)) * FFT;
    
    float avg = 0;
    for (int i = 0; i < FFT; ++i) {
        avg += Input[IDX(row_start + i, col_start + (col_id + i) % FFT)];
    }
    avg /= FFT;

    for (int i = 0; i < FFT; ++i) {
        Output[IDX(row_start + i, col_start + (col_id + i) % FFT)] = avg;      
    }

#undef IDX
}

static cudaError_t
BMC_gpu(const float *Input, float *Output, const int ROW, const int COL, const int FFT) 
{
    cudaError_t err;

    int total_thread = (ROW / FFT) * FFT * (COL / FFT);
    int nthread = props[dev_used].maxThreadsPerBlock;
    int nblock = (total_thread + nthread - 1) / nthread;

    if (err = cudaMemset(Output, 0, ROW * COL * sizeof(float)), err != cudaSuccess)
        return err;
    BMC_kernel<<<nblock, nthread>>>(Input, Output, ROW, COL, FFT);
    return cudaSuccess;
}

static void 
BMC_cpu(const float *Input, float *Output, const int ROW, const int COL, const int FFT)
{
#define IDX(_i, _j) ((_i) * COL + (_j))

    float *sub_vec = 0, *vec = 0;

    sub_vec = (float *)malloc(FFT * sizeof(float));
    vec = (float *)malloc(COL * sizeof(float));

    for (int i = 0; i < ROW / FFT; ++i) {
        memset(sub_vec, 0, FFT * sizeof(float));
        memset(vec, 0, COL * sizeof(float));

        for (int t = 0; t < COL / FFT; ++t) 
            for (int m = 0; m < FFT; ++m) 
                for (int n = 0; n < FFT; ++n) 
                    vec[t * FFT + m] += Input[IDX(FFT * i + n, t * FFT + (m + n) % FFT)] / FFT;
        
        for (int j = 0; j < FFT; ++j) {
            for (int k = 0; k < COL / FFT; ++k) 
                if (j >= 1) {
                    sub_vec[0] = vec[FFT * (k + 1) - 1];
                    memcpy(sub_vec + 1, vec + FFT * k, (FFT - 1) * sizeof(float));
                    memcpy(vec + FFT * k, sub_vec, FFT * sizeof(float));
                }
            memcpy(Output + IDX(FFT * i + j, 0), vec, COL * sizeof(float));
        }
    }

    if (sub_vec)
        free(sub_vec);
    if (vec)
        free(vec);

#undef IDX
}

static double 
eval_cpu(const int ROW, const int COL, const int FFT)
{  
    float *Input = 0, *Output = 0;
    struct timespec start, stop;
    double cost = -1;
    double acc = 0;

    Input = (float *)malloc(ROW * COL * sizeof(float));
    Output = (float *)malloc(ROW * COL * sizeof(float));

    BMC_cpu(Input, Output, ROW, COL, FFT);

    acc = 0;
    for (int i = 0; i < number; ++i) {
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        BMC_cpu(Input, Output, ROW, COL, FFT);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop);
        acc += (stop.tv_nsec - start.tv_nsec) * 1e-6;
    }

    cost = acc / number;

    if (Input)
        free(Input);
    if (Output)
        free(Output);

    return cost;   
}
static void setup_gpu()
{
    panic(cudaGetDeviceCount(&ndev));
    for (int i = 0; i < ndev; ++i) {
        panic(cudaGetDeviceProperties(props + i, i));
    }
}

static double 
eval_gpu(const int ROW, const int COL, const int FFT)
{
    float *Input = 0, *Output = 0;
    cudaEvent_t start = 0, stop = 0;
    double cost = -1;
    double acc = 0;

    fail(cudaMalloc(&Input, ROW * COL * sizeof(float)));
    fail(cudaMalloc(&Output, ROW * COL * sizeof(float)));

    // warm up
    BMC_gpu(Input, Output, ROW, COL, FFT);
    fail(cudaDeviceSynchronize());

    fail(cudaEventCreate(&start));
    fail(cudaEventCreate(&stop));

    acc = 0;
    for (int i = 0; i < number; ++i) {
        float ms = 0;
        fail(cudaEventRecord(start, 0));
        fail(BMC_gpu(Input, Output, ROW, COL, FFT));
        fail(cudaEventRecord(stop, 0));
        fail(cudaEventSynchronize(stop));
        fail(cudaEventElapsedTime(&ms, start, stop));
        acc += ms;
    }
    cost = acc / number;

cleanup: 
    if (Input)
        panic(cudaFree(Input));
    if (Output)
        panic(cudaFree(Output));
    if (start)
        panic(cudaEventDestroy(start));
    if (stop)
        panic(cudaEventDestroy(stop));

    return cost;
}

const int shapes[6][3] = {
    {1024, 256, 8},
    {1024, 256, 16},
    {1024, 512, 8},
    {1024, 512, 16},
    {1024, 40, 8},
    {1024, 40, 16},
};
const int NSHAPE = (sizeof(shapes) / sizeof(int[3]));

static void
single_test(const int ROW, const int COL, const int FFT)
{
#define IDX(_i, _j) ((_i) * COL + (_j))
    srand(time(NULL));

    float *in = 0, *out = 0;
    float *din = 0, *dout = 0;

    in = (float *)malloc(ROW * COL * sizeof(float));
    assert(in);
    out = (float *)malloc(ROW * COL * sizeof(float));    
    assert(out);

    panic(cudaMalloc(&din, ROW * COL * sizeof(float)));
    panic(cudaMalloc(&dout, ROW * COL * sizeof(float)));

    for (int i = 0; i < ROW; ++i) 
        for (int j = 0; j < COL; ++j)
            in[IDX(i, j)] = rand() * 100.0 / RAND_MAX;

    panic(cudaMemcpy(din, in, ROW * COL * sizeof(float), cudaMemcpyHostToDevice));

    BMC_cpu(in, out, ROW, COL, FFT);
    BMC_gpu(din, dout, ROW, COL, FFT);
    panic(cudaDeviceSynchronize());

    panic(cudaMemcpy(in, dout, ROW * COL * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < ROW; ++i) {
        for (int j = 0; j < COL; ++j) {
            if (fabs(in[IDX(i, j)] - out[IDX(i, j)]) > 1e-6) {
                fprintf(stderr, "shape(%d, %d, %d) different in (%d, %d): %f(GPU) vs %f(CPU)\n", ROW, COL, FFT, i, j, in[IDX(i,j)], out[IDX(i,j)]);
                goto cleanup;
            }
        } 
    }
    printf("shape(%d, %d, %d) passed\n", ROW, COL, FFT);

cleanup:
    free(in);
    free(out);
    panic(cudaFree(din));
    panic(cudaFree(dout));

#undef IDX
}

static void 
test() 
{
    printf("Start test ...\n");
    for (int i = 0; i < NSHAPE; ++i) {
        const int *s = shapes[i];
        single_test(s[0], s[1], s[2]);
    }
    printf("Done\n");
}

int main(int argc, char **argv) 
{
    double (*evaler)(const int, const int, const int) = 0;
    const char *target = 0;

    if (argc < 2) {
        goto help;
    } else {
        target = argv[1];
        if (!strcmp(target, "GPU")) {
            setup_gpu();
            evaler = eval_gpu;
        } else if (!strcmp(target, "CPU")) {
            evaler = eval_cpu;
        } else if (!strcmp(target, "Test")){
            setup_gpu();
            test();
            exit(0);
        } else {
            goto help;
        }
    }

    assert(evaler && target);
    printf("Start eval in %s ...\n", target);
    for (int i = 0; i < NSHAPE; ++i) {
        const int *s = shapes[i];
        double cost = evaler(s[0], s[1], s[2]);
        printf("Layer: %d\nTime Cost: %F ms\n", i, cost);
    }
    printf("Done\n");
    exit(0);
    
help: 
    fprintf(
        stderr,
        "Usage: %s [CPU | GPU | Test]\n",
        argv[0]
    );
    exit(1);
}