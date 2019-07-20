#ifndef MULTIGPU_CORR
#define MULTIGPU_CORR

#include <cufft.h>

typedef struct{
  cufftHandle planFFT;
  cufftHandle planFFTreversed;

  float *d_events, *d_events_reversed, *d_corr_t, *d_norms;
  cufftComplex *d_events_freq, *d_events_reversed_freq, *d_corr_f;

  float *d_xcorr_vals_pos, *d_xcorr_vals_neg;
  int *d_xcorr_lags_pos, *d_xcorr_lags_neg;

  cudaStream_t stream;

} MULTIplan;

/**
 * Function that prints all the values of the different CUDA devices, obtained
 * using the API. Also limits the number of GPUS to use, in case nGPUS (the
 * desired number of devices to use) surpasses the total number of available
 * devices.
 */
void PrintCUDA (int *nGPUS);

/**
 * Function that prints all the values received from Python on the terminal.
 */
void PrintValues (int n_events, int event_length, int shift, int paddedSize,
                  int fftsize, int nGPUS, float threshold);

/**
 * Function that computes the starting point for each device. A verbose option
 * could be included in the future
 */
void ComputeIterations (int nGPUS, int n_events, int *Iterations);

/**
 * Function that calculates how many element each device has to compute. This
 * number is extremely useful to check the load-balance.
 */
void ComputeElements (int nGPUS, int n_events, int *Iterations, unsigned *n_elements);

/**
 * Function that calculates the norm of every time-series, so the normalisation
 * of the magnitudes can be correctly done.
 */
void ComputeNorms(float *events, float *norms, int n_events, int event_length, int paddedSize);


/**
 * Function that implements what each CPU-thread has to perform before entering
 * the hotspot of the application.
 */
void GPU_Thread(int id, int *Iterations, int n_elements, float *events, float *events_reversed,
        int n_events, int event_length, int paddedSize, int fftsize, int shift,
        float threshold, float *norms, float *xcorr_vals_pos, int *xcorr_lags_pos,
        float *xcorr_vals_neg, int *xcorr_lags_neg);


/**
 * Function that implements the calls for multiplication and the IFFT and
 * FeatureExtraction functions. This is the hotspot of the program.
 */
void MultiplicationAndIFFT (int id, cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
      int n_events, int event_length, int paddedSize, int fftsize, int shift, int n_elements,
      float threshold, int beginning, int end, float *d_norms, cufftComplex *d_corr_f, float *d_corr_t,
      float *d_xcorr_vals_pos, int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg, cudaStream_t stream);


/**
 * This kernel describes how to perform the Element-Wise-Multiplication between
 * two arrays (or vectors) using one block for each one. We also asume that the
 * number of elements is a-power-of-two plus one. That is why we have to add
 * an 'if' block at the end, for the first thread.
 * This function is developed for the linearised problem.
 */
__global__ void EWM_Linear (cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
        int n_events, int fftsize, int ref_i, int ref_j, cufftComplex *d_corr_f);


/**
 * This functions implements the Feature Extraction for the linearised problem,
 * where each block of Threads will take care of one cross-correlation in time
 * domain.
 */
__global__ void FE_Linear(float *d_corr_t, int shift, int n_events,
        int paddedSize, int event_length, float *d_norms,
        int ref_i, int ref_j, float threshold, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg);


/**
 * This Feature Extraction function was developed for the 'blocked' problem,
 * where a single GPU was computing blocks of the output matrices. In this case,
 * half of the computational power was completely wasted, since the output
 * matrices are symmetric.
 */
__global__ void ElementWiseMultiplication(cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
        int n_events, int fftsize, int chunk_number, int chunkSize, cufftComplex *d_corr_f);


/**
 * This Feature Extraction function was developed for the 'blocked' problem,
 * where a single GPU was computing blocks of the output matrices. In this case,
 * half of the computational power was completely wasted, since the output
 * matrices are symmetric.
 */
__global__ void FeatureExtraction(float *d_corr_t, int shift, int n_events, int paddedSize,
        int event_length, int chunk_number, int chunkSize, float *d_norms, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg);

/**
 * Function that implements only the Element-Wise-Multiplication of the upper
 * matrix.
 */
__global__ void ElementWiseMultiplicationUpper(cufftComplex *d_events_freq, cufftComplex *d_events_reversed_freq,
        int n_events, int fftsize, int chunk_number, int chunkSize, cufftComplex *d_corr_f);


/**
 * Function that implements only the Feature Extraction of the upper matrix.
 */
__global__ void FeatureExtractionUpper(float *d_corr_t, int shift, int n_events, int paddedSize,
        int event_length, int chunk_number, float *d_norms, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg);


/**
 * This is a beta-function. It means that this function was created only with
 * the purpose of checking the values obtained from GPU, at the beginning of
 * the development. No performance goal is intended.
 */
__global__ void FeatureExtraction1TH(float *d_corr_t, int shift, int n_events, int paddedSize,
        int event_length, int chunk_number, int chunkSize, float *d_norms, float *d_xcorr_vals_pos,
        int *d_xcorr_lags_pos, float *d_xcorr_vals_neg, int *d_xcorr_lags_neg);


/**
 * Function that implements the Feature Extraction in CPU. Could be useful in
 * the future if we want to implement an heterogeneous version of the
 * application.
 */
void FeatureExtractionCPU (float *corr_t, int shift, int event_length, int paddedSize,
        float &norm1, float &norm2, float &val_pos, int &lag_pos, float &val_neg, int &lag_neg);


#endif
