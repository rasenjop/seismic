#include <stdio.h>
#include <cufft.h>
#include <chrono>
#include <iostream>
#include <omp.h>
#include <cmath>

#define NUM_THREADS 128
#define BS 15000//20000

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

// ElementWiseMultiplication for linearised problem
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

// FeatureExtraction function for linearised problem
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
    if(shared_min_val[tid] / (d_norms[i] * d_norms[j] * paddedSize) < -threshold){
      d_xcorr_vals_neg[i*n_events - i*(i-1)/2 + j - i] = shared_min_val[tid] / (d_norms[i] * d_norms[j] * paddedSize);
      d_xcorr_lags_neg[i*n_events - i*(i-1)/2 + j - i] = shared_min_lag[tid];
    }
  }
}

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



void MultiplicationAndIFFT (cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
      int n_events, int event_length, int paddedSize, int fftsize, int shift, int n_elements,
      float threshold, float *d_norms, cufftComplex *d_corr_f, float *d_corr_t,
      float *d_xcorr_vals_pos, int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg){

  int num_chunks = n_elements / BS;
  printf("The number of chunks is: %d\n", num_chunks);
  printf("The size of every chunk is: %d\n", BS);

  int ref_i = 0;
  int ref_j = 0;
  int leftovers = BS;

  int n[1] = {paddedSize};

  cufftHandle planIFFT;
  cufftPlanMany(&planIFFT, 1, n,
              NULL, 1, fftsize,
              NULL, 1, paddedSize,
              CUFFT_C2R, BS);

  dim3 gridDim = {unsigned(BS), unsigned(1), 1U};


  for(int k=0; k<num_chunks; k++){
    // printf("Iteration %d\n", k);
    // printf("La referencia es: {%d, %d}\n", ref_i, ref_j);
    // getchar();
    // auto time1=std::chrono::high_resolution_clock::now();

    EWM_Linear<<<gridDim, 512>>> (d_events_freq, d_events_reversed_freq, n_events,
                      fftsize, ref_i, ref_j, d_corr_f);
    cudaDeviceSynchronize();

    // auto time2=std::chrono::high_resolution_clock::now();

    // auto time3=std::chrono::high_resolution_clock::now();
    // for (int q=0; q<chunkSize/num_rows_ifft; q++){
    //   cufftExecC2R (planIFFT, &d_corr_f[q * num_rows_ifft * n_events * fftsize],
    //                           &d_corr_t[q * num_rows_ifft * n_events * paddedSize]);
    // }
    cufftExecC2R (planIFFT, d_corr_f, d_corr_t);

    // auto time4=std::chrono::high_resolution_clock::now();

    // auto time5=std::chrono::high_resolution_clock::now();
    FE_Linear<<<gridDim, NUM_THREADS>>> (d_corr_t, shift, n_events, paddedSize,
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

extern "C"{
  //Function used to initialise the CUDA RunTime
  void initialiseCUDA(){
    float *d_example;
    cufftComplex *d_result;
    int number = 128;
    int fftsize = number/2 + 1;

    cufftHandle planFFT;

    cudaMalloc((void**) &d_example, sizeof(float) * number);
    cudaMalloc((void**) &d_result, sizeof(cufftComplex) * fftsize);

    int n[1] = {number};

    cufftPlanMany(&planFFT, 1, n,
                NULL, 1, number,
                NULL, 1, fftsize,
                CUFFT_R2C, 1);
    cufftExecR2C(planFFT, d_example, d_result);
  }


  void correlationCUDA(float *events, float *events_reversed , int n_events, int event_length,
                      int shift, int paddedSize, int num_threads, float threshold,
                      float *xcorr_vals_pos, int *xcorr_lags_pos,
                      float *xcorr_vals_neg, int *xcorr_lags_neg){
    // int nDevices;
    //
    // cudaGetDeviceCount(&nDevices);
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

    printf("\n------------------Values received from Python------------------\n");
    int fftsize = paddedSize / 2 + 1;
    printf("C: n_events: %d\n", n_events);
    printf("C: event_length: %d\n", event_length);
    printf("C: shift: %d\n", shift);
    printf("C: paddedSize: %d\n", paddedSize);
    printf("C: fftsize: %d\n", fftsize);
    printf("C: num_threads: %d\n", num_threads);
    printf("C: threshold: %f\n", threshold);


    // Hardcoded variables
    //int chunkSize = 1309440 / 2; //numero de correlaciones cruzadas que caben en la memoria global
    // entre dos porque tenemos la correlacion en tiempo y en frecuencia
    // chunkSize /= n_events;
    // int chunkSize = 1;
    // printf("C: El tamaño del chunk es: %d\n", chunkSize);
    unsigned n_elements = unsigned(n_events * (n_events+1) / 2);

    cufftHandle planFFT;
    cufftHandle planFFTreversed;

    float *d_events, *d_events_reversed, *d_corr_t, *d_norms;
    cufftComplex *d_events_freq, *d_events_reversed_freq, *d_corr_f;

    float *d_xcorr_vals_pos, *d_xcorr_vals_neg;
    int *d_xcorr_lags_pos, *d_xcorr_lags_neg;

    // A cufftComplex is composed by 2 floats, weighing 8 bytes in total
    // Memory reserve for events in time domain and copy from host
    cudaMalloc((void**) &d_events, sizeof(float) * n_events * paddedSize);
    cudaMemcpy(d_events, events, sizeof(float) * n_events * paddedSize, cudaMemcpyHostToDevice);

    cudaMalloc((void**) &d_events_reversed, sizeof(float) * n_events * paddedSize);
    cudaMemcpy(d_events_reversed, events_reversed, sizeof(float) * n_events * paddedSize, cudaMemcpyHostToDevice);

    // Memory reserve for FFTs
    cudaMalloc((void**) &d_events_freq, sizeof(cufftComplex) * n_events * fftsize);
    cudaMalloc((void**) &d_events_reversed_freq, sizeof(cufftComplex) * n_events * fftsize);

    // Memory reserve for correlations in frequency and time domain
    cudaMalloc((void**) &d_corr_f, sizeof(cufftComplex) * BS * fftsize);
    cudaMalloc((void**) &d_corr_t, sizeof(float) * BS * paddedSize);

    // Memory reserve for output matrices
    cudaMalloc((void**) &d_xcorr_vals_pos, sizeof(float) * n_elements);
    cudaMalloc((void**) &d_xcorr_vals_neg, sizeof(float) * n_elements);
    cudaMalloc((void**) &d_xcorr_lags_pos, sizeof(int) * n_elements);
    cudaMalloc((void**) &d_xcorr_lags_neg, sizeof(int) * n_elements);


    float norms[n_events];
    ComputeNorms(events, norms, n_events, event_length, paddedSize);
    cudaMalloc((void**) &d_norms, sizeof(float) * n_events);
    cudaMemcpy(d_norms, norms, sizeof(float) * n_events, cudaMemcpyHostToDevice);

    // THE FFT COMPUTATION STARTS
    int n[1] = {paddedSize};
    // Cabecera de la funcion: cufftExecR2C(cufftHandle plan, cufftReal *idata, cufftComplex *odata);

    auto time1=std::chrono::high_resolution_clock::now();
    cufftPlanMany(&planFFT, 1, n,
                NULL, 1, paddedSize,
                NULL, 1, fftsize,
                CUFFT_R2C, n_events);

    cufftExecR2C(planFFT, d_events, d_events_freq);

    auto time2=std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo FFT events: " << std::chrono::duration<double>(time2-time1).count() << std::endl;


    auto time3 = std::chrono::high_resolution_clock::now();
    cufftPlanMany(&planFFTreversed, 1, n,
                NULL, 1, paddedSize,
                NULL, 1, fftsize,
                CUFFT_R2C, n_events);

    cufftExecR2C(planFFTreversed, d_events_reversed, d_events_reversed_freq);

    auto time4 = std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo FFT events_reversed: " << std::chrono::duration<double>(time4-time3).count() << std::endl;

    cudaFree(d_events);
    cudaFree(d_events_reversed);


    MultiplicationAndIFFT(d_events_freq, d_events_reversed_freq, n_events, event_length,
            paddedSize, fftsize, shift, n_elements, threshold, d_norms, d_corr_f, d_corr_t,
            d_xcorr_vals_pos, d_xcorr_lags_pos, d_xcorr_vals_neg, d_xcorr_lags_neg);
    auto time7 = std::chrono::high_resolution_clock::now();

    printf("Copying memory starts:\n");
    cudaMemcpy (xcorr_vals_pos, d_xcorr_vals_pos, sizeof(float) * n_elements, cudaMemcpyDeviceToHost);
    cudaMemcpy (xcorr_vals_neg, d_xcorr_vals_neg, sizeof(float) * n_elements, cudaMemcpyDeviceToHost);
    cudaMemcpy (xcorr_lags_pos, d_xcorr_lags_pos, sizeof(int) * n_elements, cudaMemcpyDeviceToHost);
    cudaMemcpy (xcorr_lags_neg, d_xcorr_lags_neg, sizeof(int) * n_elements, cudaMemcpyDeviceToHost);

    auto time8 = std::chrono::high_resolution_clock::now();
    std::cout << "Tiempo copia memoria: " << std::chrono::duration<double>(time8-time7).count() << std::endl;
    //cudaMemcpy(events_reversed_freq, d_events_reversed_freq, sizeof(cufftComplex) * n_events * fftsize, cudaMemcpyDeviceToHost);


    cudaFree(d_corr_t);
    cudaFree(d_norms);
    cudaFree(d_events_freq);
    cudaFree(d_events_reversed_freq);
    cudaFree(d_corr_f);
    cudaFree(d_xcorr_vals_pos);
    cudaFree(d_xcorr_vals_neg);
    cudaFree(d_xcorr_lags_pos);
    cudaFree(d_xcorr_lags_neg);


    printf("C: Finished computing in the GPU\n");
  }
}
