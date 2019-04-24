#include "correlation_c.h"
#include <vector>
#include <fftw3.h>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iostream>

//#include "extrae_user_events.h"

using namespace std; //es necesario? ¿Por qué?

/*Tags empleados:
- TODO: por implementar o cambios a realizar
- PROBLEMA
- DUDA
*/


float normalize(fftwf_complex* signal1_t, fftwf_complex* signal2_t, int event_length){ //normalizes a pair of signals
  float pot_1 = 0.0, pot_2 = 0.0;

  for(int i=0; i<event_length; i++){
    pot_1 += signal1_t[i][0]*signal1_t[i][0];
    pot_2 += signal2_t[i][0]*signal2_t[i][0];
  }
  return sqrt(pot_1*pot_2);
}



void ComputeFFT(float* signals_t, float* signals_t_reversed, fftwf_complex* signals_f, fftwf_complex* signals_f_reversed,
                int n_signals, int paddedSize, int fftsize){

  fftwf_plan plan;
  for(int i=0; i<n_signals; i++){
    plan = fftwf_plan_dft_r2c_1d(paddedSize, &signals_t[i*paddedSize], &signals_f[i*fftsize], FFTW_ESTIMATE);
    //p = fftwf_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);

    fftwf_execute(plan);

    plan = fftwf_plan_dft_r2c_1d(paddedSize, &signals_t_reversed[i*paddedSize], &signals_f_reversed[i*fftsize], FFTW_ESTIMATE);
    fftwf_execute(plan);
  }
  fftwf_destroy_plan(plan);
}


//FUNCIÓN PARA CALCULAR EL VALOR ABSOLUTO
  int av(int number){
    return (number<0)? -number:number;
  }


void inverseFFT(fftwf_complex *corr_f, float *corr_t, fftwf_plan plan, int event_length,
                int shift, int paddedSize, float &norm1, float &norm2, float threshold,
                int *priv_max_hist, int *priv_min_hist, int n_classes_hist, float delta_hist,
                float &val_pos, int &lag_pos, float &val_neg, int &lag_neg){

  // auto time1=std::chrono::high_resolution_clock::now();
  fftwf_execute(plan);
  // auto time2=std::chrono::high_resolution_clock::now();
  // std::cout << "Tiempo IFFT: " << std::chrono::duration<double,std::milli>(time2-time1).count() << std::endl;

  // auto time3=std::chrono::high_resolution_clock::now();
  float pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  // float av_max = 0.0; //máximo valor absoluto obtenido


  for(int i = 0; i < shift; i++){
    if(corr_t[i]>pos){
      pos = corr_t[i];
      l_pos = i + 1;
    }
    else if(corr_t[i]<neg){
      neg = corr_t[i];
      l_neg = i + 1;
    }
  }

  for(int i = event_length - shift; i <= event_length; i++){
    if(corr_t[i] > pos){
      pos = corr_t[i];
      l_pos = i - event_length + 1;
    }
    else if(corr_t[i]<neg){
      neg = corr_t[i];
      l_neg = i - event_length + 1;
    }
  }


  //av_max = (abs(pos)>abs(neg))? pos:abs(neg);
  val_pos = pos / ((float) paddedSize * norm1 * norm2);
  if(val_pos > 0.9999999) priv_max_hist[n_classes_hist-1]++; //TODO: never noone talked about it again
  else  priv_max_hist[int((val_pos)/delta_hist)]++;

  if(val_pos > threshold){
    lag_pos = l_pos;
  }
  else{
    val_pos = 0.0;
    lag_pos = 0;
  }

  val_neg = neg / ((float) paddedSize * norm1 * norm2);
  //priv_min_hist[int(-val_neg/delta_hist)]++;
  if(val_neg < -0.9999999) priv_min_hist[n_classes_hist-1]++; //TODO: never noone talked about it again
  else  priv_min_hist[int(-(val_neg)/delta_hist)]++;
  if(val_neg < -threshold){
    lag_neg = l_neg;
  }
  else{
    val_neg = 0.0;
    lag_neg = 0;
  }


  // auto time4=std::chrono::high_resolution_clock::now();
  // std::cout << "Tiempo Search: " << std::chrono::duration<double,std::milli>(time4-time3).count() << std::endl;
}

void sweep(fftw_complex *xcorrij, float &val_pos, int &lag_pos, float &val_neg, int &lag_neg, int paddedSize){
  //en realidad deberían ser de tipo int, al menos los lags, pero no sé si puede haber problemas al pasar las variables a python
  //además, hay que calcular los lags relativos al centro de la función.
  float pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  for(int i=0; i<paddedSize; i++){
    if(xcorrij[i][0]>pos){
      pos = xcorrij[i][0];
      l_pos = i;
    }
    else if(xcorrij[i][0]<neg){
      neg = xcorrij[i][0];
      l_neg = i;
    }
  }

  val_pos = pos;
  lag_pos = l_pos;
  val_neg = neg;
  lag_neg = l_neg;
}




//En caso de no poder utilizar vectores, se creara una funcion para invertir las señales
//tambien podria generarse una funcion que, en lugar de hacer la convolucion, hiciese
//la correlacion directamente.
void reverseArray(fftwf_complex *signal, int size){
  fftwf_complex aux;
  int half = size/2;

  //Por ejemplo, si son 5, el cociente entre 5 y 2 son 2, por lo que se recorrerian
  //los dos primeros elementos. Esto es perfecto, porque no haría falta pasar por
  //el elemento central

  for(int i=0; i<half; i++){ //me da un poco de miedo el size/2
    aux[0] = signal[i][0];
    aux[1] = signal[i][1];

    signal[i][0] = signal[size-i][0];
    signal[i][1] = signal[size-i][1];

    signal[size-i][0] = aux[0];
    signal[size-i][i] = aux[1];
  }
}

void ElementWiseMultiplication(fftwf_complex *signal_a, fftwf_complex *signal_b, fftwf_complex *result, int fftsize){
  for(int i=0; i<fftsize; i++){
    result[i][0] = signal_a[i][0] * signal_b[i][0] - signal_a[i][1] * signal_b[i][1];
    result[i][1] = signal_a[i][1] * signal_b[i][0] + signal_a[i][0] * signal_b[i][1];
  }
}

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


//Funcion que implementa la llamada propiamente dicha, haciendo uso del resto de funciones
//arriba declaradas
extern "C"{
  void correlationCPP(float *events, float *events_reversed , int n_events, int event_length,
                      int shift, int paddedSize, int num_threads, int chunk_size, float threshold,
                      int n_classes_hist, float *xcorr_vals_pos, int *xcorr_lags_pos,
                      float *xcorr_vals_neg, int *xcorr_lags_neg, int *max_hist, int *min_hist){
    /*
    events:   conjunto de señales en el tiempo y con zero-appended
    events_reversed: conjunto de señales en el tiempo reversed y zero-appended
    n_events: len(events)    -- numero de series temporales en events
    length_event:            -- tamaño de los datos de una serie temporal
    shift:       int         -- 2*shift+1 será el tamaño de la correlacion resultante (con ind=0 en el punto central)
    */

    printf("------------------Values received from Python------------------\n");
    printf("C: n_events: %d\n", n_events);
    printf("C: event_length: %d\n", event_length);
    printf("C: shift: %d\n", shift);
    printf("C: paddedSize: %d\n", paddedSize);
    int fftsize = paddedSize / 2 + 1;
    printf("C: fftsize: %d\n", fftsize);
    float delta_hist = float(1.0/n_classes_hist);
    printf("C: delta_hist is %f\n\n", delta_hist);

    fftwf_complex* signals_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) *(n_events * fftsize));
    fftwf_complex* signals_reversed_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) *(n_events * fftsize));
    printf("C: FFT memory successfully allocated\n");

    float norms[n_events];
    printf("C: Norms memory successfully allocated\n");



    // auto time1 = std::chrono::high_resolution_clock::now();

    ComputeNorms(events, norms, n_events, event_length, paddedSize);

    // auto time2=std::chrono::high_resolution_clock::now();
    // std::cout << "Tiempo norms: " << std::chrono::duration<double,std::milli>(time2-time1).count() << std::endl;
    printf("C: Norms have been computed\n");


    // auto time3=std::chrono::high_resolution_clock::now();
    ComputeFFT(events, events_reversed, signals_freq, signals_reversed_freq, n_events, paddedSize, fftsize);

    // auto time4=std::chrono::high_resolution_clock::now();
    // std::cout << "Tiempo FFTs: " << std::chrono::duration<double,std::milli>(time4-time3).count() << std::endl;

    printf("C: FFTs have been computed\n\n");

    printf("C: The number of threads is %d\n", num_threads);
    printf("C: The chunk_size is %d\n", chunk_size);
    printf("C: Feature extraction begins..\n");

    omp_set_num_threads(num_threads);
    #pragma omp parallel default(shared)
    {
      int base_index = 0;
      int* priv_max_hist = (int*) calloc (n_classes_hist, sizeof(int));
      int* priv_min_hist = (int*) calloc (n_classes_hist, sizeof(int));
      fftwf_complex* corr_f = (fftwf_complex*) fftwf_malloc (sizeof(fftwf_complex) * fftsize);
      float* corr_t = (float*) fftwf_malloc (sizeof(float) * paddedSize);
      fftwf_plan plan;
      #pragma omp critical
      plan = fftwf_plan_dft_c2r_1d(paddedSize, corr_f, corr_t, FFTW_MEASURE);
      #pragma omp for schedule(dynamic, chunk_size)
      for(int i=0; i<n_events; i++){
        for(int j=0; j<n_events-i; j++){ //tengo que mirar como hace scipy la correlacion
          // auto time5=std::chrono::high_resolution_clock::now();
          base_index = i*n_events - i*(i-1)/2;
          ElementWiseMultiplication(&signals_freq[i*fftsize], &signals_reversed_freq[(i+j)*fftsize], corr_f, fftsize);
          // auto time6=std::chrono::high_resolution_clock::now();
          // std::cout << "Tiempo ElementWiseMult: " << std::chrono::duration<double,std::milli>(time6-time5).count() << std::endl;

          // auto time7=std::chrono::high_resolution_clock::now();
          inverseFFT(corr_f, corr_t, plan, event_length, shift, paddedSize, norms[i], norms[i+j],
                     threshold, priv_max_hist, priv_min_hist, n_classes_hist, delta_hist,
                     xcorr_vals_pos[base_index + j], xcorr_lags_pos[base_index + j],
                     xcorr_vals_neg[base_index + j], xcorr_lags_neg[base_index + j]);
          // auto time8=std::chrono::high_resolution_clock::now();
          // std::cout << "Tiempo IFFT + Search: " << std::chrono::duration<double,std::milli>(time8-time7).count() << std::endl;
        }
      }
      #pragma omp critical
      for(int i = 0; i < n_classes_hist; i++){
        max_hist[i] += priv_max_hist[i];
        min_hist[i] += priv_min_hist[i];
      }
      fftwf_free(corr_f);
      //fftwf_free(corr_t);
      fftwf_destroy_plan(plan);

    }

    printf("C: Finished computing the features!\n");

    fftwf_cleanup();
  }
}
