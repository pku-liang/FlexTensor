/*
 copy from https://github.com/pyrovski/cublasSgemmBatched-example/blob/master/gemm.cpp
*/
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

int main(int argc, char ** argv){

  int status;
  int lower = 0;
  int upper = 1;
  int num = 25000;
  int reps = 5;
  int verbose = 0;
  int M = 100;
  int N = 20;
  int K = 50;
  int B = 4;
  
  while((status = getopt(argc, argv, "M:N:K:B:r:v")) != -1){
    switch(status){
    case 'M':
      M = strtoul(optarg, 0, 0);
      break;
    case 'N':
      N = strtoul(optarg, 0, 0);
      break;
    case 'K':
      K = strtoul(optarg, 0, 0);
      break;
    case 'B':
      B = strtoul(optarg, 0, 0);
      break;
    case 'r':
      reps = strtoul(optarg, 0, 0);
      break;
    case 'v':
      verbose = 1;
      break;
    default:
      cerr << "invalid argument: " << status << endl;
      exit(1);
    }
  }
  if(verbose) 
    cout << "running with" 
	 << " lower: " << lower
	 << " upper: " << upper
	 << " num: " << num
	 << " reps: " << reps
     << endl;

  num = B;

  if(verbose) cout << "initializing inputs" << endl;
  float *matrices = (float*)malloc(M * K * num * sizeof(float));
  float *vectors = (float*)malloc(N * K * num * sizeof(float));

  assert(matrices);
  assert(vectors);

  for(int i = 0; i < num * K * M; i++)
    matrices[i] = drand48();

  for(int i = 0; i < num * K * N; i++)
    vectors[i] = drand48();

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS){
    cerr << "cublas init failed" << endl;
    exit(1);
  }

  if(verbose) cout << "allocating device variables" << endl;

  // allocate input space on device
  float *devMatrices;
  size_t devMatricesPitch;
  cudaStat = 
    cudaMallocPitch(&devMatrices,
		    &devMatricesPitch,
		    M * sizeof(float),
		    num * K);

  assert(!cudaStat);

  float *devVectors = 0;
  size_t devVectorsPitch;
  cudaStat = 
    cudaMallocPitch(&devVectors,
		    &devVectorsPitch,
		    K * sizeof(float),
		    num * N);

  assert(!cudaStat);

  // allocate result space on device
  float *devResult = 0;
  size_t devResultPitch;
  cudaStat = 
    cudaMallocPitch(&devResult,
		    &devResultPitch,
		    M * sizeof(float),
		    num * N);

  assert(!cudaStat);

  if(verbose) cout << "copying data to device" << endl;
  // copy data to device
  cudaStat = 
    cudaMemcpy2D(devMatrices,
		 devMatricesPitch,
		 matrices,
		 M * sizeof(float),
		 M * sizeof(float),
		 K * num,
		 cudaMemcpyHostToDevice);

  assert(!cudaStat);
  
  cudaStat = 
    cudaMemcpy2D(devVectors,
		 devVectorsPitch,
		 vectors,
		 K * sizeof(float),
		 K * sizeof(float),
		 num * N,
		 cudaMemcpyHostToDevice);

  assert(!cudaStat);

  // create lists of device pointers to inputs and outputs
  float **AList = 0, **BList = 0, **CList = 0;

  AList = (float**)malloc(num * sizeof(float*));
  BList = (float**)malloc(num * sizeof(float*));
  CList = (float**)malloc(num * sizeof(float*));

  for(int i = 0; i < num; i++){
    AList[i] = devMatrices + devMatricesPitch/sizeof(float) * K * i;
    BList[i] = devVectors + devVectorsPitch/sizeof(float) * i * N;
    CList[i] = devResult + devResultPitch/sizeof(float) * i * N;
  }

  // copy pointer lists to device
  float **devAList = 0, **devBList = 0, **devCList = 0;
  cudaStat = cudaMalloc(&devAList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMalloc(&devBList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMalloc(&devCList, num * sizeof(float*));
  assert(!cudaStat);

  cudaStat = cudaMemcpy(devAList,
			AList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  assert(!cudaStat);
  
  cudaStat = cudaMemcpy(devBList,
			BList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  assert(!cudaStat);

  cudaStat = cudaMemcpy(devCList,
			CList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  assert(!cudaStat);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  int 
    lda = devMatricesPitch / sizeof(float),
    ldb = devVectorsPitch / sizeof(float),
    ldc = devResultPitch / sizeof(float);
  const float alpha = 1.0f, beta = 0.0f;

  /* perform <num> <size x size> x <size x 1> multiplications 
     with distinct matrices
   */
  for(int size = lower; size < upper; size++){
    // if(verbose) cout << "running with size " << size << endl;
    double sum = 0.0;
    for(int rep = 0; rep < reps; rep++){
      cudaEventRecord(start, 0);
      stat = cublasSgemmBatched(handle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				M,
				N,
				K,
				&alpha,
				(const float**)devAList,
				lda,
				(const float**)devBList,
				ldb,
				&beta,
				devCList,
				ldc,
				num);
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      if(stat != CUBLAS_STATUS_SUCCESS){
	cerr << "cublasSgemmBatched failed" << endl;
	exit(1);
      }
      assert(!cudaGetLastError());
      
      float elapsed;
      cudaEventElapsedTime(&elapsed, start, stop);
      sum += elapsed;
      
    //   if(verbose)
	// cout << "distinct; size " << size << ": " << elapsed << " s; " 
	//      << elapsed / num << " s per operation" << endl;
    }
    cout << "(" << M << "," << N << "," << K << "," << B << ")" << " repeat " << reps << " average: " << sum/reps << " ms; "
	 << sum / reps / num << " ms per maxtrix" << endl;
  }

  /* Perform <num> <size x size> x <size x 1> multiplications 
     with a single matrix.
     Is it possible to use constant memory cublas?
  */

//   for(int i = 0; i < num; i++)
//     AList[i] = devMatrices;

//   for(int size = lower; size <= upper; size++){
//     if(verbose) cout << "running with size " << size << endl;
//     double sum = 0.0;
//     for(int rep = 0; rep < reps; rep++){
//       cudaEventRecord(start, 0);
//       stat = cublasSgemmBatched(handle,
// 				CUBLAS_OP_N,
// 				CUBLAS_OP_N,
// 				size,
// 				1,
// 				size,
// 				&alpha,
// 				(const float**)devAList,
// 				lda,
// 				(const float**)devBList,
// 				ldb,
// 				&beta,
// 				devCList,
// 				ldc,
// 				num);
//       cudaEventRecord(stop,0);
//       cudaEventSynchronize(stop);
//       if(stat != CUBLAS_STATUS_SUCCESS){
// 	cerr << "cublasSgemmBatched failed" << endl;
// 	exit(1);
//       }
//       assert(!cudaGetLastError());
      
//       float elapsed;
//       cudaEventElapsedTime(&elapsed, start, stop);
//       elapsed /= 1000.0f;
//       sum += elapsed;
      
//       if(verbose)
// 	cout << "single; size " << size << ": " << elapsed << " s; " 
// 	     << elapsed / num << " s per operation" << endl;
//     }
//     cout << "single; size " << size << " average: " << sum/reps << " s; "
// 	 << sum / reps / num << " s per operation" << endl;
//   }

  free(matrices);
  free(vectors);
      
  return 0;
}