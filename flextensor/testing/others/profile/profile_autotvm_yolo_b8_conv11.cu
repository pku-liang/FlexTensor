#include "cuda_runtime.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


__global__ void conv2d( float* __restrict__ data,  float* __restrict__ kernel,  float* __restrict__ compute) {
    float compute_local[16];
   __shared__ float pad_temp_shared[112];
   __shared__ float kernel_shared[512];
   compute_local[0] = 0.000000e+00f;
   compute_local[8] = 0.000000e+00f;
   compute_local[4] = 0.000000e+00f;
   compute_local[12] = 0.000000e+00f;
   compute_local[1] = 0.000000e+00f;
   compute_local[9] = 0.000000e+00f;
   compute_local[5] = 0.000000e+00f;
   compute_local[13] = 0.000000e+00f;
   compute_local[2] = 0.000000e+00f;
   compute_local[10] = 0.000000e+00f;
   compute_local[6] = 0.000000e+00f;
   compute_local[14] = 0.000000e+00f;
   compute_local[3] = 0.000000e+00f;
   compute_local[11] = 0.000000e+00f;
   compute_local[7] = 0.000000e+00f;
   compute_local[15] = 0.000000e+00f;
   for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
     __syncthreads();
     if ((((int)threadIdx.z) * 7) < (112 - ((int)threadIdx.x))) {
       if (((int)threadIdx.x) < 7) {
         if ((((int)blockIdx.z) / 4) < (8 - (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 112))) {
           pad_temp_shared[(((((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 112) * 112) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) % 14)) + (((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 14) % 2) * 14)) + (((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 28) % 4) * 28))] = data[((((((((((int)blockIdx.z) / 4) * 200704) + (((int)blockIdx.y) * 28)) + (rc_outer * 784)) + ((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 112) * 200704)) + (((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) % 14)) + (((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 14) % 2) * 14)) + (((((((int)threadIdx.z) * 7) + ((int)threadIdx.x)) / 28) % 4) * 196))];
         }
       }
     }
     if ((((int)threadIdx.z) * 8) < (128 - ((((int)threadIdx.x) * 3) / 4))) {
       if ((((int)threadIdx.z) * 32) < (512 - (((int)threadIdx.x) * 3))) {
         if (((int)threadIdx.x) < 11) {
           if (((((int)blockIdx.z) % 4) * 128) < ((512 - (((int)threadIdx.z) * 8)) - ((((int)threadIdx.x) * 3) / 4))) {
             kernel_shared[(((((int)threadIdx.z) * 32) + (((((int)threadIdx.x) * 3) / 4) * 4)) + ((((int)threadIdx.x) * 3) % 4))] = kernel[((((((((int)blockIdx.z) % 4) * 131072) + (rc_outer * 4)) + (((int)threadIdx.z) * 8192)) + (((((int)threadIdx.x) * 3) / 4) * 1024)) + ((((int)threadIdx.x) * 3) % 4))];
           }
         }
       }
     }
     if ((((int)threadIdx.z) * 8) < (128 - ((1 + (((int)threadIdx.x) * 3)) / 4))) {
       if ((((int)threadIdx.z) * 32) < (511 - (((int)threadIdx.x) * 3))) {
         if (((int)threadIdx.x) < 11) {
           if (((((int)blockIdx.z) % 4) * 128) < ((512 - (((int)threadIdx.z) * 8)) - ((1 + (((int)threadIdx.x) * 3)) / 4))) {
             kernel_shared[(((((int)threadIdx.z) * 32) + (((1 + (((int)threadIdx.x) * 3)) / 4) * 4)) + ((1 + (((int)threadIdx.x) * 3)) % 4))] = kernel[((((((((int)blockIdx.z) % 4) * 131072) + (rc_outer * 4)) + (((int)threadIdx.z) * 8192)) + (((1 + (((int)threadIdx.x) * 3)) / 4) * 1024)) + ((1 + (((int)threadIdx.x) * 3)) % 4))];
           }
         }
       }
     }
     if ((((int)threadIdx.z) * 8) < (128 - ((2 + (((int)threadIdx.x) * 3)) / 4))) {
       if ((((int)threadIdx.z) * 32) < (510 - (((int)threadIdx.x) * 3))) {
         if (((int)threadIdx.x) < 10) {
           if (((((int)blockIdx.z) % 4) * 128) < ((512 - (((int)threadIdx.z) * 8)) - ((2 + (((int)threadIdx.x) * 3)) / 4))) {
             kernel_shared[(((((int)threadIdx.z) * 32) + (((2 + (((int)threadIdx.x) * 3)) / 4) * 4)) + ((2 + (((int)threadIdx.x) * 3)) % 4))] = kernel[((((((((int)blockIdx.z) % 4) * 131072) + (rc_outer * 4)) + (((int)threadIdx.z) * 8192)) + (((2 + (((int)threadIdx.x) * 3)) / 4) * 1024)) + ((2 + (((int)threadIdx.x) * 3)) % 4))];
           }
         }
       }
     }
     __syncthreads();
     compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(((int)threadIdx.z) * 16)]));
     compute_local[8] = (compute_local[8] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(256 + (((int)threadIdx.z) * 16))]));
     compute_local[4] = (compute_local[4] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(((int)threadIdx.z) * 16)]));
     compute_local[12] = (compute_local[12] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(256 + (((int)threadIdx.z) * 16))]));
     compute_local[1] = (compute_local[1] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(4 + (((int)threadIdx.z) * 16))]));
     compute_local[9] = (compute_local[9] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(260 + (((int)threadIdx.z) * 16))]));
     compute_local[5] = (compute_local[5] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(4 + (((int)threadIdx.z) * 16))]));
     compute_local[13] = (compute_local[13] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(260 + (((int)threadIdx.z) * 16))]));
     compute_local[2] = (compute_local[2] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(8 + (((int)threadIdx.z) * 16))]));
     compute_local[10] = (compute_local[10] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(264 + (((int)threadIdx.z) * 16))]));
     compute_local[6] = (compute_local[6] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(8 + (((int)threadIdx.z) * 16))]));
     compute_local[14] = (compute_local[14] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(264 + (((int)threadIdx.z) * 16))]));
     compute_local[3] = (compute_local[3] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(12 + (((int)threadIdx.z) * 16))]));
     compute_local[11] = (compute_local[11] + (pad_temp_shared[((int)threadIdx.x)] * kernel_shared[(268 + (((int)threadIdx.z) * 16))]));
     compute_local[7] = (compute_local[7] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(12 + (((int)threadIdx.z) * 16))]));
     compute_local[15] = (compute_local[15] + (pad_temp_shared[(14 + ((int)threadIdx.x))] * kernel_shared[(268 + (((int)threadIdx.z) * 16))]));
     compute_local[0] = (compute_local[0] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(1 + (((int)threadIdx.z) * 16))]));
     compute_local[8] = (compute_local[8] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(257 + (((int)threadIdx.z) * 16))]));
     compute_local[4] = (compute_local[4] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(1 + (((int)threadIdx.z) * 16))]));
     compute_local[12] = (compute_local[12] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(257 + (((int)threadIdx.z) * 16))]));
     compute_local[1] = (compute_local[1] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(5 + (((int)threadIdx.z) * 16))]));
     compute_local[9] = (compute_local[9] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(261 + (((int)threadIdx.z) * 16))]));
     compute_local[5] = (compute_local[5] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(5 + (((int)threadIdx.z) * 16))]));
     compute_local[13] = (compute_local[13] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(261 + (((int)threadIdx.z) * 16))]));
     compute_local[2] = (compute_local[2] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(9 + (((int)threadIdx.z) * 16))]));
     compute_local[10] = (compute_local[10] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(265 + (((int)threadIdx.z) * 16))]));
     compute_local[6] = (compute_local[6] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(9 + (((int)threadIdx.z) * 16))]));
     compute_local[14] = (compute_local[14] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(265 + (((int)threadIdx.z) * 16))]));
     compute_local[3] = (compute_local[3] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(13 + (((int)threadIdx.z) * 16))]));
     compute_local[11] = (compute_local[11] + (pad_temp_shared[(28 + ((int)threadIdx.x))] * kernel_shared[(269 + (((int)threadIdx.z) * 16))]));
     compute_local[7] = (compute_local[7] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(13 + (((int)threadIdx.z) * 16))]));
     compute_local[15] = (compute_local[15] + (pad_temp_shared[(42 + ((int)threadIdx.x))] * kernel_shared[(269 + (((int)threadIdx.z) * 16))]));
     compute_local[0] = (compute_local[0] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(2 + (((int)threadIdx.z) * 16))]));
     compute_local[8] = (compute_local[8] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(258 + (((int)threadIdx.z) * 16))]));
     compute_local[4] = (compute_local[4] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(2 + (((int)threadIdx.z) * 16))]));
     compute_local[12] = (compute_local[12] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(258 + (((int)threadIdx.z) * 16))]));
     compute_local[1] = (compute_local[1] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(6 + (((int)threadIdx.z) * 16))]));
     compute_local[9] = (compute_local[9] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(262 + (((int)threadIdx.z) * 16))]));
     compute_local[5] = (compute_local[5] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(6 + (((int)threadIdx.z) * 16))]));
     compute_local[13] = (compute_local[13] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(262 + (((int)threadIdx.z) * 16))]));
     compute_local[2] = (compute_local[2] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(10 + (((int)threadIdx.z) * 16))]));
     compute_local[10] = (compute_local[10] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(266 + (((int)threadIdx.z) * 16))]));
     compute_local[6] = (compute_local[6] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(10 + (((int)threadIdx.z) * 16))]));
     compute_local[14] = (compute_local[14] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(266 + (((int)threadIdx.z) * 16))]));
     compute_local[3] = (compute_local[3] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(14 + (((int)threadIdx.z) * 16))]));
     compute_local[11] = (compute_local[11] + (pad_temp_shared[(56 + ((int)threadIdx.x))] * kernel_shared[(270 + (((int)threadIdx.z) * 16))]));
     compute_local[7] = (compute_local[7] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(14 + (((int)threadIdx.z) * 16))]));
     compute_local[15] = (compute_local[15] + (pad_temp_shared[(70 + ((int)threadIdx.x))] * kernel_shared[(270 + (((int)threadIdx.z) * 16))]));
     compute_local[0] = (compute_local[0] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(3 + (((int)threadIdx.z) * 16))]));
     compute_local[8] = (compute_local[8] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(259 + (((int)threadIdx.z) * 16))]));
     compute_local[4] = (compute_local[4] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(3 + (((int)threadIdx.z) * 16))]));
     compute_local[12] = (compute_local[12] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(259 + (((int)threadIdx.z) * 16))]));
     compute_local[1] = (compute_local[1] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(7 + (((int)threadIdx.z) * 16))]));
     compute_local[9] = (compute_local[9] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(263 + (((int)threadIdx.z) * 16))]));
     compute_local[5] = (compute_local[5] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(7 + (((int)threadIdx.z) * 16))]));
     compute_local[13] = (compute_local[13] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(263 + (((int)threadIdx.z) * 16))]));
     compute_local[2] = (compute_local[2] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(11 + (((int)threadIdx.z) * 16))]));
     compute_local[10] = (compute_local[10] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(267 + (((int)threadIdx.z) * 16))]));
     compute_local[6] = (compute_local[6] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(11 + (((int)threadIdx.z) * 16))]));
     compute_local[14] = (compute_local[14] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(267 + (((int)threadIdx.z) * 16))]));
     compute_local[3] = (compute_local[3] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(15 + (((int)threadIdx.z) * 16))]));
     compute_local[11] = (compute_local[11] + (pad_temp_shared[(84 + ((int)threadIdx.x))] * kernel_shared[(271 + (((int)threadIdx.z) * 16))]));
     compute_local[7] = (compute_local[7] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(15 + (((int)threadIdx.z) * 16))]));
     compute_local[15] = (compute_local[15] + (pad_temp_shared[(98 + ((int)threadIdx.x))] * kernel_shared[(271 + (((int)threadIdx.z) * 16))]));
   }
   compute[((((((int)blockIdx.z) * 25088) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[0];
   compute[((((12544 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[8];
   compute[((((14 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[4];
   compute[((((12558 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[12];
   compute[((((196 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[1];
   compute[((((12740 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[9];
   compute[((((210 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[5];
   compute[((((12754 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[13];
   compute[((((392 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[2];
   compute[((((12936 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[10];
   compute[((((406 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[6];
   compute[((((12950 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[14];
   compute[((((588 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[3];
   compute[((((13132 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[11];
   compute[((((602 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[7];
   compute[((((13146 + (((int)blockIdx.z) * 25088)) + (((int)blockIdx.y) * 28)) + (((int)threadIdx.z) * 784)) + ((int)threadIdx.x))] = compute_local[15];
 }
 

 float random_float(int a = 1)
{
    return (float)rand() / (float)(RAND_MAX / a);
}


int main()
{
    srand(time(NULL));
    int batch = 8;
    int CI = 1024;
    int CO = 512;
    int Height = 14;
    int Width = 14;
    int kernel = 1;

    int Image_size = batch * CI * Height * Width;
    float *Image = new float[Image_size];
    int W_size = CO * CI * kernel * kernel;
    float *W = new float[W_size];
    int Output_size = batch * CO * Height * Width;
    float *Output = new float[Output_size];

    float *dImage, *dW, *dOutput;
    
    // initialize
    for (int i = 0; i < Image_size; ++i)
        Image[i] = random_float();
    for (int i = 0; i < W_size; ++i)
        W[i] = random_float();
    for (int i = 0; i < Output_size; ++i)
        Output[i] = 0;
    
    cudaMalloc((void**)&dImage, Image_size * sizeof(float));
    cudaMalloc((void**)&dW, W_size * sizeof(float));
    cudaMalloc((void**)&dOutput, Output_size * sizeof(float));

    cudaMemcpy(dImage, Image, Image_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W, W_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dOutput, Output, Output_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(1, 7, 32);
    dim3 block(14, 1, 16);
    conv2d<<< grid, block >>>(dImage, dW, dOutput);

    cudaMemcpy(Output, dOutput, Output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dImage);
    cudaFree(dW);
    cudaFree(dOutput);
    delete [] Image;
    delete [] W;
    delete [] Output;
    return 0;
}