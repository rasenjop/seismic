#include <stdio.h>
#include <cufft.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>
#include <vector>
#include <thread>

#include "correlation_multicuda.h"

#define NUM_THREADS 128
#define BS 7000//20000

using namespace std;

extern "C"{
  //Function used to initialise the CUDA RunTime
  void initialiseCUDA(int nGPUS){
    int number = 128;
    int fftsize = number/2 + 1;

    int n[1] = {number};

    cufftHandle planFFT;

    for(int i=0; i<nGPUS; i++){
      cudaSetDevice(i);
      float *d_example;
      cufftComplex *d_result;

      cudaMalloc((void**) &d_example, sizeof(float) * number);
      cudaMalloc((void**) &d_result, sizeof(cufftComplex) * fftsize);

      cufftPlanMany(&planFFT, 1, n,
                  NULL, 1, number,
                  NULL, 1, fftsize,
                  CUFFT_R2C, 1);
      cufftExecR2C(planFFT, d_example, d_result);

      cudaFree(d_example);
      cudaFree(d_result);
    }
  }



  void correlationCUDA(float *events, float *events_reversed , int n_events, int event_length,
                      int shift, int paddedSize, float threshold, int nGPUS,
                      float *xcorr_vals_pos, int *xcorr_lags_pos,
                      float *xcorr_vals_neg, int *xcorr_lags_neg){

    PrintCUDA(&nGPUS);

    int fftsize = paddedSize / 2 + 1;
    PrintValues (n_events, event_length, shift, paddedSize, fftsize, nGPUS,
      threshold);

    int *Iterations = new int[nGPUS+1];
    for(int i=0; i<=nGPUS; i++) Iterations[i] = 0;
    ComputeIterations (nGPUS, n_events, Iterations);


    // Compute the norms for all the input Time-Series
    float *norms = new float[n_events];
    ComputeNorms(events, norms, n_events, event_length, paddedSize);


    // Let's find out how many elements each gpu has to compute
    unsigned *n_elements = new unsigned[nGPUS];
    ComputeElements (nGPUS, n_events, Iterations, n_elements);

    vector<thread> threads;
    int i;
    int acc = 0;
    for (i=0; i<nGPUS; i++) {
      if(i==0){
        threads.push_back(thread(GPU_Thread, i, Iterations, n_elements[i], events,
                    events_reversed, n_events, event_length,
                    paddedSize, fftsize, shift, threshold, norms,
                    xcorr_vals_pos, xcorr_lags_pos,
                    xcorr_vals_neg, xcorr_lags_neg));
      }
      else{
        acc += n_elements[i-1];
        threads.push_back(thread(GPU_Thread, i, Iterations, n_elements[i], &events[Iterations[i] * paddedSize],
                    &events_reversed[Iterations[i] * paddedSize], n_events - Iterations[i], event_length,
                    paddedSize, fftsize, shift, threshold, &norms[Iterations[i]],
                    &xcorr_vals_pos[acc], &xcorr_lags_pos[acc],
                    &xcorr_vals_neg[acc], &xcorr_lags_neg[acc]));
      }
    }
    for (auto &t: threads)
      t.join();

    printf("C: Finished computing in the GPUS\n\n");
  }
}



/**
 * Function that prints all the values received from Python on the terminal.
 */
void PrintValues (int n_events, int event_length, int shift, int paddedSize,
  int fftsize, int nGPUS, float threshold){

  printf("\n------------------Values received from Python------------------\n");
  printf("C: n_events: %d\n", n_events);
  printf("C: event_length: %d\n", event_length);
  printf("C: shift: %d\n", shift);
  printf("C: paddedSize: %d\n", paddedSize);
  printf("C: fftsize: %d\n", fftsize);
  printf("C: nGPUS: %d\n", nGPUS);
  printf("C: threshold: %f\n\n", threshold);
}


/**
 * Function that prints all the values of the different CUDA devices, obtained
 * using the API. Also limits the number of GPUS to use, in case nGPUS (the
 * desired number of devices to use) surpasses the total number of available
 * devices.
 */
void PrintCUDA(int *nGPUS){
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  if(*nGPUS > nDevices) *nGPUS = nDevices;

  // for (int i = 0; i < nDevices; i++) {
  //   cudaDeviceProp prop;
  //   cudaGetDeviceProperties(&prop, i);
  //   printf("Device Number: %d\n", i);
  //   printf("  Device name: %s\n", prop.name);
  //   printf("  Memory Clock Rate (KHz): %d\n",
  //          prop.memoryClockRate);
  //   printf("  Memory Bus Width (bits): %d\n",
  //          prop.memoryBusWidth);
  //   printf("  Peak Memory Bandwidth (GB/s): %f\n",
  //          2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  //   printf("  Max Threads per Block: %d\n",
  //          prop.maxThreadsPerBlock);
  //   printf("  Max Threads Per MultiProcessor: %d\n",
  //          prop.maxThreadsPerMultiProcessor);
  //   printf("  Max Thread Dim: %d %d %d\n",
  //          prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  //   printf("  Max Grid Size: %d %d %d\n\n",
  //          prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  // }
}

/**
 * Function that computes the starting point for each device. A verbose option
 * could be included in the future
 */
void ComputeIterations (int nGPUS, int n_events, int *Iterations){
  int acc = 0;
  for(int i=nGPUS; i>0; i--){
    acc = 0;
    if(i==nGPUS){
      Iterations[i] = int(n_events / sqrt(nGPUS));
    }
    else{
      for(int j=nGPUS; j>i; j--){
        acc += Iterations[j];
      }
      Iterations[i] = int(- acc + sqrt((acc * acc) +  (n_events * n_events) / nGPUS));
    }
  }

  for(int i=1; i<=nGPUS; i++){
    Iterations[i] += Iterations[i-1];
  }
  Iterations[nGPUS] = n_events;

  // printf("Valores de inicio de cada bloque de trabajo para %d devices:\n", nGPUS);
  // for(int i=0; i<=nGPUS; i++){
  //   printf("Para el device %d - Fila %d\n", i, Iterations[i]);
  // }
}


/**
 * Function that calculates how many element each device has to compute. This
 * number is extremely useful to check the load-balance.
 */
void ComputeElements (int nGPUS, int n_events, int *Iterations, unsigned *n_elements){
  int acc = 0;
  for(int i=nGPUS-1; i>=0; i--){
    n_elements[i] = (n_events - Iterations[i]) * (n_events - Iterations[i] + 1) / 2 - acc;
    acc += n_elements[i];
  }

  // printf("Número de elementos para cada GPU:\n");
  // for(int i=0; i<nGPUS; i++){
  //   printf("Para el device %d - %d elementos\n", i, n_elements[i]);
  // }

  // int elementos = n_events * (n_events+1) / 2;
  // printf("El total de elementos iniciales es: %d\n", elementos);
  // printf("El número de elementos computados es: %d\n", acc);
}


/**
 * Function that calculates the norm of every time-series, so the normalisation
 * of the magnitudes can be correctly done.
 */
void ComputeNorms(float *events, float *norms, int n_events, int event_length, int paddedSize){
  float pot =0.0;
  for(int i=0; i<n_events; i++){
    pot=0.0;
    for(int j=0; j<event_length; j++){
      pot += events[i*paddedSize+j] * events[i*paddedSize+j];
    }
    norms[i] = sqrt(pot);
  }
}


/**
 * Function that implements what each CPU-thread has to perform before entering
 * the hotspot of the application.
 */
void GPU_Thread(int id, int *Iterations, int n_elements, float *events, float *events_reversed,
        int n_events, int event_length, int paddedSize, int fftsize, int shift,
        float threshold, float *norms, float *xcorr_vals_pos, int *xcorr_lags_pos,
        float *xcorr_vals_neg, int *xcorr_lags_neg){

  MULTIplan plan;

  cudaSetDevice(id);
  cudaStreamCreate(&plan.stream);

  cudaMalloc((void**) &plan.d_events, sizeof(float) * n_events * paddedSize);
  cudaMemcpy(plan.d_events, events, sizeof(float) * n_events * paddedSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &plan.d_events_reversed, sizeof(float) * n_events * paddedSize);
  cudaMemcpy(plan.d_events_reversed, events_reversed, sizeof(float) * n_events * paddedSize, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &plan.d_norms, sizeof(float) * n_events);
  cudaMemcpy(plan.d_norms, norms, sizeof(float) * n_events, cudaMemcpyHostToDevice);

  cudaMalloc((void**) &plan.d_events_freq, sizeof(cufftComplex) * n_events * fftsize);
  cudaMalloc((void**) &plan.d_events_reversed_freq, sizeof(cufftComplex) * n_events * fftsize);

  cudaMalloc((void**) &plan.d_corr_f, sizeof(cufftComplex) * BS * fftsize);
  cudaMalloc((void**) &plan.d_corr_t, sizeof(float) * BS * paddedSize);

  cudaMalloc((void**) &plan.d_xcorr_vals_pos, sizeof(float) * n_elements);
  cudaMalloc((void**) &plan.d_xcorr_vals_neg, sizeof(float) * n_elements);
  cudaMalloc((void**) &plan.d_xcorr_lags_pos, sizeof(int) * n_elements);
  cudaMalloc((void**) &plan.d_xcorr_lags_neg, sizeof(int) * n_elements);

  int n[1] = {paddedSize};

  auto time1=std::chrono::high_resolution_clock::now();
  cufftPlanMany(&plan.planFFT, 1, n,
              NULL, 1, paddedSize,
              NULL, 1, fftsize,
              CUFFT_R2C, n_events);

  cufftExecR2C(plan.planFFT, plan.d_events, plan.d_events_freq);

  auto time2=std::chrono::high_resolution_clock::now();
  std::cout << "Thread " << id << " - Tiempo FFT events: " << std::chrono::duration<double>(time2-time1).count() << std::endl;


  auto time3 = std::chrono::high_resolution_clock::now();
  cufftPlanMany(&plan.planFFTreversed, 1, n,
              NULL, 1, paddedSize,
              NULL, 1, fftsize,
              CUFFT_R2C, n_events);

  cufftExecR2C(plan.planFFTreversed, plan.d_events_reversed, plan.d_events_reversed_freq);

  auto time4 = std::chrono::high_resolution_clock::now();
  std::cout << "Thread " << id << " - Tiempo FFT events_reversed: " << std::chrono::duration<double>(time4-time3).count() << std::endl;

  cudaFree(plan.d_events);
  cudaFree(plan.d_events_reversed);


  MultiplicationAndIFFT(id, plan.d_events_freq, plan.d_events_reversed_freq, n_events, event_length,
          paddedSize, fftsize, shift, n_elements, threshold, Iterations[id], Iterations[id+1],
          plan.d_norms, plan.d_corr_f, plan.d_corr_t, plan.d_xcorr_vals_pos,
          plan.d_xcorr_lags_pos, plan.d_xcorr_vals_neg, plan.d_xcorr_lags_neg, plan.stream);
  auto time7 = std::chrono::high_resolution_clock::now();

  printf("Thread %d - Copying memory starts:\n", id);
  cudaMemcpy (xcorr_vals_pos, plan.d_xcorr_vals_pos, sizeof(float) * n_elements, cudaMemcpyDeviceToHost);
  cudaMemcpy (xcorr_vals_neg, plan.d_xcorr_vals_neg, sizeof(float) * n_elements, cudaMemcpyDeviceToHost);
  cudaMemcpy (xcorr_lags_pos, plan.d_xcorr_lags_pos, sizeof(int) * n_elements, cudaMemcpyDeviceToHost);
  cudaMemcpy (xcorr_lags_neg, plan.d_xcorr_lags_neg, sizeof(int) * n_elements, cudaMemcpyDeviceToHost);

  auto time8 = std::chrono::high_resolution_clock::now();
  std::cout << "Thread " << id << " - Tiempo copia memoria: " << std::chrono::duration<double>(time8-time7).count() << std::endl;

  cudaFree(plan.d_corr_t);
  cudaFree(plan.d_norms);
  cudaFree(plan.d_events_freq);
  cudaFree(plan.d_events_reversed_freq);
  cudaFree(plan.d_corr_f);
  cudaFree(plan.d_xcorr_vals_pos);
  cudaFree(plan.d_xcorr_vals_neg);
  cudaFree(plan.d_xcorr_lags_pos);
  cudaFree(plan.d_xcorr_lags_neg);
}


/**
 * Function that implements the calls for multiplication and the IFFT and
 * FeatureExtraction functions. This is the hotspot of the program.
 */
void MultiplicationAndIFFT (int id, cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
      int n_events, int event_length, int paddedSize, int fftsize, int shift, int n_elements,
      float threshold, int beginning, int end, float *d_norms, cufftComplex *d_corr_f, float *d_corr_t,
      float *d_xcorr_vals_pos, int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg, cudaStream_t stream){

  int num_chunks = n_elements / BS;

  int ref_i = 0;
  int ref_j = ref_i;
  int leftovers = BS;

  int n[1] = {paddedSize};

  cufftHandle planIFFT;
  cufftPlanMany(&planIFFT, 1, n,
              NULL, 1, fftsize,
              NULL, 1, paddedSize,
              CUFFT_C2R, BS);

  dim3 gridDim = {unsigned(BS), unsigned(1), 1U};


  for(int k=0; k<num_chunks; k++){
    // auto time1=std::chrono::high_resolution_clock::now();

    EWM_Linear<<<gridDim, 256, 0, stream>>> (d_events_freq, d_events_reversed_freq, n_events,
                      fftsize, ref_i, ref_j, d_corr_f);
    cudaDeviceSynchronize();
    // auto time2=std::chrono::high_resolution_clock::now();

    // auto time3=std::chrono::high_resolution_clock::now();
    cufftExecC2R (planIFFT, d_corr_f, d_corr_t);
    // auto time4=std::chrono::high_resolution_clock::now();

    // auto time5=std::chrono::high_resolution_clock::now();
    FE_Linear<<<gridDim, NUM_THREADS, 0, stream>>> (d_corr_t, shift, n_events, paddedSize,
                      event_length, d_norms, ref_i, ref_j, threshold, d_xcorr_vals_pos,
                      d_xcorr_lags_pos, d_xcorr_vals_neg, d_xcorr_lags_neg);
    cudaDeviceSynchronize();
    // auto time6=std::chrono::high_resolution_clock::now();

    // std::cout << "Tiempo FeatureExtraction events: " << std::chrono::duration<double>(time6-time5).count() << std::endl;
    // std::cout << "Iteration " << k << " {Mult, IFFT, FeatExtr} -- {" << std::chrono::duration<double>(time2-time1).count() << ", " <<
    //              std::chrono::duration<double>(time4-time3).count() << ", " << std::chrono::duration<double>(time6-time5).count() <<
    //              "}" << std::endl;

    // Update the reference
    leftovers = BS;
    while(leftovers > 0){

      if(leftovers - (n_events - ref_j) >= 0){
        leftovers -= (n_events - ref_j);
        ref_i++;
        ref_j = ref_i;
      }

      else{
        ref_j += leftovers;
        leftovers = 0;
      }
    }//while
  }//chunk for

  // Computing the leftovers at the end of the triangular matrices
  leftovers = n_elements - num_chunks * BS;

  if(leftovers > 0){
    // printf("Thread %d - La referencia para los leftovers(%d) es: {%d, %d}\n", id, leftovers, ref_i, ref_j);

    cufftPlanMany(&planIFFT, 1, n,
                NULL, 1, fftsize,
                NULL, 1, paddedSize,
                CUFFT_C2R, leftovers);

    dim3 gridDim = {unsigned(leftovers), unsigned(1), 1U};

    EWM_Linear<<<gridDim, 512>>> (d_events_freq, d_events_reversed_freq, n_events,
                      fftsize, ref_i, ref_j, d_corr_f);
    cudaDeviceSynchronize();

    cufftExecC2R (planIFFT, d_corr_f, d_corr_t);

    FE_Linear<<<gridDim, NUM_THREADS>>> (d_corr_t, shift, n_events, paddedSize,
                      event_length, d_norms, ref_i, ref_j, threshold, d_xcorr_vals_pos,
                      d_xcorr_lags_pos, d_xcorr_vals_neg, d_xcorr_lags_neg);
    cudaDeviceSynchronize();
  }
}//function



/**
 * This kernel describes how to perform the Element-Wise-Multiplication between
 * two arrays (or vectors) using one block for each one. We also asume that the
 * number of elements is a-power-of-two plus one. That is why we have to add
 * an 'if' block at the end, for the first thread.
 * This function is developed for the linearised problem.
 */
__global__ void EWM_Linear (cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
        int n_events, int fftsize, int ref_i, int ref_j, cufftComplex *d_corr_f){

  int tid = threadIdx.x;
  int i, j;

  i = ref_i;
  j = ref_j;
  int leftovers = blockIdx.x;

  while(leftovers > 0){

    if(leftovers - (n_events - j) >= 0){
      leftovers -= (n_events - j);
      i++;
      j = i;
    }

    else{
      j += leftovers;
      leftovers = 0;
    }
  }

  //Create a loop where every thread computes several multiplications of the same xcorr
  for(int k=0; k<int(fftsize/blockDim.x); k++){
    d_corr_f[blockIdx.x * fftsize + tid + k * blockDim.x].x =
                             d_events_freq[i * fftsize + tid + k * blockDim.x].x
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].x
                           - d_events_freq[i * fftsize + tid + k * blockDim.x].y
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].y;

    d_corr_f[blockIdx.x * fftsize + tid + k * blockDim.x].y =
                             d_events_freq[i * fftsize + tid + k * blockDim.x].y
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].x
                           + d_events_freq[i * fftsize + tid + k * blockDim.x].x
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].y;
  }

  if (tid == 0){
    d_corr_f[(blockIdx.x+1) * fftsize - 1].x =
                             d_events_freq[(i+1) * fftsize - 1].x
                           * d_events_reversed_freq[(j+1) * fftsize - 1].x
                           - d_events_freq[(i+1) * fftsize - 1].y
                           * d_events_reversed_freq[(j+1) * fftsize - 1].y;

    d_corr_f[(blockIdx.x+1) * fftsize - 1].y =
                             d_events_freq[(i+1) * fftsize - 1].y
                           * d_events_reversed_freq[(j+1) * fftsize - 1].x
                           + d_events_freq[(i+1) * fftsize - 1].x
                           * d_events_reversed_freq[(j+1) * fftsize - 1].y;
  }
}

/**
 * This functions implements the Feature Extraction for the linearised problem,
 * where each block of Threads will take care of one cross-correlation in time
 * domain.
 */
__global__ void FE_Linear(float *d_corr_t, int shift, int n_events,
        int paddedSize, int event_length, float *d_norms,
        int ref_i, int ref_j, float threshold, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg){

  int tid = threadIdx.x;
  int i, j;

  i = ref_i;
  j = ref_j;
  int leftovers = blockIdx.x;

  while(leftovers > 0){

    if(leftovers - (n_events - j) >= 0){
      leftovers -= (n_events - j);
      i++;
      j = i;
    }

    else{
      j += leftovers;
      leftovers = 0;
    }
  }

  __shared__ float shared_max_val[NUM_THREADS];
  __shared__ int shared_max_lag[NUM_THREADS];
  __shared__ float shared_min_val[NUM_THREADS];
  __shared__ int shared_min_lag[NUM_THREADS];

  shared_max_val[tid] = 0.0;
  shared_max_lag[tid] = 0;
  shared_min_val[tid] = 0.0;
  shared_min_lag[tid] = 0;

  for(int k = -int(shift/blockDim.x); k<int(shift/blockDim.x); k++){ //generalise for the number of threads per block
    if(d_corr_t[blockIdx.x * paddedSize + event_length + k * blockDim.x + tid] > shared_max_val[tid]){
      shared_max_val[tid] = d_corr_t[blockIdx.x * paddedSize + event_length + k * blockDim.x + tid];
      shared_max_lag[tid] = k * blockDim.x + tid + 1;
    }
    else if(d_corr_t[blockIdx.x * paddedSize + event_length + k * blockDim.x + tid] < shared_min_val[tid]){
      shared_min_val[tid] = d_corr_t[blockIdx.x * paddedSize + event_length + k * blockDim.x + tid];
      shared_min_lag[tid] = k * blockDim.x + tid + 1;
    }
    __syncthreads();
  }

  for (int stride = blockDim.x/2; stride>0; stride>>=1){
    if(tid < stride){
      if(shared_max_val[tid] < shared_max_val[tid + stride]){
        shared_max_val[tid] = shared_max_val[tid + stride];
        shared_max_lag[tid] = shared_max_lag[tid + stride];
      }

      if(shared_min_val[tid] > shared_min_val[tid + stride]){
        shared_min_val[tid] = shared_min_val[tid + stride];
        shared_min_lag[tid] = shared_min_lag[tid + stride];
      }
    }
    __syncthreads();
  }

  // Data will be written onto the output matrices only if the threshold is surpassed
  if(tid == 0){
    if(shared_max_val[tid] / (d_norms[i] * d_norms[j] * paddedSize) > threshold){
      d_xcorr_vals_pos[i*n_events - i*(i-1)/2 + j - i] = shared_max_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
      d_xcorr_lags_pos[i*n_events - i*(i-1)/2 + j - i] = shared_max_lag[tid];
    }
    else{
      d_xcorr_vals_pos[i*n_events - i*(i-1)/2 + j - i] = 0.0;
      d_xcorr_lags_pos[i*n_events - i*(i-1)/2 + j - i] = 0;
    }

    if(shared_min_val[tid] / (d_norms[i] * d_norms[j] * paddedSize) < -threshold){
      d_xcorr_vals_neg[i*n_events - i*(i-1)/2 + j - i] = shared_min_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
      d_xcorr_lags_neg[i*n_events - i*(i-1)/2 + j - i] = shared_min_lag[tid];
    }
    else{
      d_xcorr_vals_neg[i*n_events - i*(i-1)/2 + j - i] = 0.0;
      d_xcorr_lags_neg[i*n_events - i*(i-1)/2 + j - i] = 0;
    }
  }
}


/**
 * This Feature Extraction function was developed for the 'blocked' problem,
 * where a single GPU was computing blocks of the output matrices. In this case,
 * half of the computational power was completely wasted, since the output
 * matrices are symmetric.
 */
__global__ void ElementWiseMultiplication(cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
        int n_events, int fftsize, int chunk_number, int chunkSize, cufftComplex *d_corr_f){
  int tid = threadIdx.x;
  int i, j;
  //int block = blockIdx.x % chunkSize;
  i = blockIdx.y + chunk_number * chunkSize;
  j = blockIdx.x;

  //Create a loop where every thread computes several multiplications of the same xcorr
  for(int k=0; k<int(fftsize/blockDim.x); k++){
    d_corr_f[(blockIdx.y * n_events + blockIdx.x) * fftsize + tid + k * blockDim.x].x =
                             d_events_freq[i * fftsize + tid + k * blockDim.x].x
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].x
                           - d_events_freq[i * fftsize + tid + k * blockDim.x].y
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].y;

    d_corr_f[(blockIdx.y * n_events + blockIdx.x) * fftsize + tid + k * blockDim.x].y =
                             d_events_freq[i * fftsize + tid + k * blockDim.x].y
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].x
                           + d_events_freq[i * fftsize + tid + k * blockDim.x].x
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].y;
  }

  if (tid == 0){
    d_corr_f[(blockIdx.y * n_events + blockIdx.x + 1) * fftsize - 1].x =
                             d_events_freq[(i+1) * fftsize - 1].x
                           * d_events_reversed_freq[(j+1) * fftsize - 1].x
                           - d_events_freq[(i+1) * fftsize - 1].y
                           * d_events_reversed_freq[(j+1) * fftsize -1].y;

    d_corr_f[(blockIdx.y * n_events + blockIdx.x + 1) * fftsize - 1].y =
                             d_events_freq[(i+1) * fftsize - 1].y
                           * d_events_reversed_freq[(j+1) * fftsize - 1].x
                           + d_events_freq[(i+1) * fftsize - 1].x
                           * d_events_reversed_freq[(j+1) * fftsize -1].y;
  }
}


/**
 * This Feature Extraction function was developed for the 'blocked' problem,
 * where a single GPU was computing blocks of the output matrices. In this case,
 * half of the computational power was completely wasted, since the output
 * matrices are symmetric.
 */
__global__ void FeatureExtraction(float *d_corr_t, int shift, int n_events, int paddedSize,
        int event_length, int chunk_number, int chunkSize, float *d_norms, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg){

  int tid = threadIdx.x;
  int i, j;
  // int block = blockIdx.x % chunkSize;
  i = blockIdx.y + chunk_number * chunkSize;
  j = blockIdx.x;

  __shared__ float shared_max_val[NUM_THREADS];
  __shared__ int shared_max_lag[NUM_THREADS];
  __shared__ float shared_min_val[NUM_THREADS];
  __shared__ int shared_min_lag[NUM_THREADS];

  shared_max_val[tid] = 0.0;
  shared_max_lag[tid] = 0;
  shared_min_val[tid] = 0.0;
  shared_min_lag[tid] = 0;

  for(int k = -int(shift/blockDim.x); k<int(shift/blockDim.x); k++){ //generalise for the number of threads per block
    if(d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k * blockDim.x + tid] > shared_max_val[tid]){
      shared_max_val[tid] = d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k * blockDim.x + tid];
      shared_max_lag[tid] = k * blockDim.x + tid + 1;
    }
    else if(d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k * blockDim.x + tid] < shared_min_val[tid]){
      shared_min_val[tid] = d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k * blockDim.x + tid];
      shared_min_lag[tid] = k * blockDim.x + tid + 1;
    }
    __syncthreads();
  }

  for (int stride = blockDim.x/2; stride>0; stride>>=1){
    if(tid < stride){
      if(shared_max_val[tid] < shared_max_val[tid + stride]){
        shared_max_val[tid] = shared_max_val[tid + stride];
        shared_max_lag[tid] = shared_max_lag[tid + stride];
      }

      if(shared_min_val[tid] > shared_min_val[tid + stride]){
        shared_min_val[tid] = shared_min_val[tid + stride];
        shared_min_lag[tid] = shared_min_lag[tid + stride];
      }
    }
    __syncthreads();
  }

  if(tid == 0){
    d_xcorr_vals_pos[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = shared_max_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
    d_xcorr_vals_neg[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = shared_min_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
    d_xcorr_lags_pos[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = shared_max_lag[tid];
    d_xcorr_lags_neg[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = shared_min_lag[tid];
  }
}


/**
 * Function that implements only the Element-Wise-Multiplication of the upper
 * matrix.
 */
__global__ void ElementWiseMultiplicationUpper(cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
        int n_events, int fftsize, int chunk_number, int chunkSize, cufftComplex *d_corr_f){
  int tid = threadIdx.x;
  int i, j;
  //int block = blockIdx.x % chunkSize;
  i = chunk_number;
  j = blockIdx.x + chunk_number;

  //Create a loop where every thread computes several multiplications of the same xcorr
  for(int k=0; k<int(fftsize/blockDim.x); k++){
    d_corr_f[j * fftsize + tid + k * blockDim.x].x =
                             d_events_freq[i * fftsize + tid + k * blockDim.x].x
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].x
                           - d_events_freq[i * fftsize + tid + k * blockDim.x].y
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].y;

    d_corr_f[j * fftsize + tid + k * blockDim.x].y =
                             d_events_freq[i * fftsize + tid + k * blockDim.x].y
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].x
                           + d_events_freq[i * fftsize + tid + k * blockDim.x].x
                           * d_events_reversed_freq[j * fftsize + tid + k * blockDim.x].y;
  }

  if (tid == 0){
    d_corr_f[(j+1) * fftsize - 1].x =
                             d_events_freq[(i+1) * fftsize - 1].x
                           * d_events_reversed_freq[(j+1) * fftsize - 1].x
                           - d_events_freq[(i+1) * fftsize - 1].y
                           * d_events_reversed_freq[(j+1) * fftsize -1].y;

    d_corr_f[(j+1) * fftsize - 1].y =
                             d_events_freq[(i+1) * fftsize - 1].y
                           * d_events_reversed_freq[(j+1) * fftsize - 1].x
                           + d_events_freq[(i+1) * fftsize - 1].x
                           * d_events_reversed_freq[(j+1) * fftsize -1].y;
  }
}

/**
 * Function that implements only the Feature Extraction of the upper matrix.
 */
__global__ void FeatureExtractionUpper(float *d_corr_t, int shift, int n_events, int paddedSize,
        int event_length, int chunk_number, float *d_norms, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg){

  int tid = threadIdx.x;
  int i, j;

  i = chunk_number;
  j = blockIdx.x + chunk_number;

  __shared__ float shared_max_val[NUM_THREADS];
  __shared__ int shared_max_lag[NUM_THREADS];
  __shared__ float shared_min_val[NUM_THREADS];
  __shared__ int shared_min_lag[NUM_THREADS];

  shared_max_val[tid] = 0.0;
  shared_max_lag[tid] = 0;
  shared_min_val[tid] = 0.0;
  shared_min_lag[tid] = 0;

  for(int k = -int(shift/blockDim.x); k<int(shift/blockDim.x); k++){ //generalise for the number of threads per block
    if(d_corr_t[j * paddedSize + event_length + k * blockDim.x + tid] > shared_max_val[tid]){
      shared_max_val[tid] = d_corr_t[j * paddedSize + event_length + k * blockDim.x + tid];
      shared_max_lag[tid] = k * blockDim.x + tid + 1;
    }
    else if(d_corr_t[j * paddedSize + event_length + k * blockDim.x + tid] < shared_min_val[tid]){
      shared_min_val[tid] = d_corr_t[j * paddedSize + event_length + k * blockDim.x + tid];
      shared_min_lag[tid] = k * blockDim.x + tid + 1;
    }
    __syncthreads();
  }

  for (int stride = blockDim.x/2; stride>0; stride>>=1){
    if(tid < stride){
      if(shared_max_val[tid] < shared_max_val[tid + stride]){
        shared_max_val[tid] = shared_max_val[tid + stride];
        shared_max_lag[tid] = shared_max_lag[tid + stride];
      }

      if(shared_min_val[tid] > shared_min_val[tid + stride]){
        shared_min_val[tid] = shared_min_val[tid + stride];
        shared_min_lag[tid] = shared_min_lag[tid + stride];
      }
    }
    __syncthreads();
  }

  if(tid == 0){
    d_xcorr_vals_pos[i * n_events + j] = shared_max_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
    d_xcorr_vals_neg[i * n_events + j] = shared_min_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
    d_xcorr_lags_pos[i * n_events + j] = shared_max_lag[tid];
    d_xcorr_lags_neg[i * n_events + j] = shared_min_lag[tid];
  }
}


/**
 * This is a beta-function. It means that this function was created only with
 * the purpose of checking the values obtained from GPU, at the beginning of
 * the development. No performance goal is intended.
 */
__global__ void FeatureExtraction1TH(float *d_corr_t, int shift, int n_events, int paddedSize,
        int event_length, int chunk_number, int chunkSize, float *d_norms, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg){

  int tid = threadIdx.x;

  if(tid == 0){
    int i, j;
    i = blockIdx.y + chunk_number * chunkSize;
    j = blockIdx.x;

    float val_pos = 0.0; float val_neg = 0.0;
    int lag_pos = 0;     int lag_neg = 0;

    for(int k = -shift; k < shift; k++){
      if(d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k]>val_pos){
        val_pos = d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k];
        lag_pos = k + 1;
      }
      else if(d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k]<val_neg){
        val_neg = d_corr_t[(blockIdx.y * n_events + blockIdx.x) * paddedSize + event_length + k];
        lag_neg = k + 1;
      }
    }
    val_pos /= ((float) paddedSize * d_norms[i] * d_norms[j]);
    val_neg /= ((float) paddedSize * d_norms[i] * d_norms[j]);

    d_xcorr_vals_pos[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = val_pos;
    d_xcorr_vals_neg[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = val_neg;
    d_xcorr_lags_pos[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = lag_pos;
    d_xcorr_lags_neg[(blockIdx.y + chunk_number * chunkSize) * n_events + blockIdx.x] = lag_neg;
  }
}


/**
 * Function that implements the Feature Extraction in CPU. Could be useful in
 * the future if we want to implement an heterogeneous version of the
 * application.
 */
void FeatureExtractionCPU (float *corr_t, int shift, int event_length, int paddedSize,
        float &norm1, float &norm2, float &val_pos, int &lag_pos, float &val_neg, int &lag_neg){

  val_pos = 0.0; val_neg = 0.0;
  lag_pos = 0;   lag_neg = 0;

  for(int i = 0; i < shift; i++){
    if(corr_t[i]>val_pos){
      val_pos = corr_t[i];
      lag_pos = i + 1;
    }
    else if(corr_t[i]<val_neg){
      val_neg = corr_t[i];
      lag_neg = i + 1;
    }
  }

  for(int i = event_length - shift; i <= event_length; i++){
    if(corr_t[i] > val_pos){
      val_pos = corr_t[i];
      lag_pos = i - event_length + 1;
    }
    else if(corr_t[i]<val_neg){
      val_neg = corr_t[i];
      lag_neg = i - event_length + 1;
    }
  }
  val_pos /= ((float) paddedSize * norm1 * norm2);
  val_neg /= ((float) paddedSize * norm1 * norm2);

}
