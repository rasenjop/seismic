#include "correlation_c.h"
#include <vector>
#include <fftw3.h>
#include <cmath>
#include <omp.h>

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



//aqui la señal llega ya recortada (250 muestras) y en el dominio del tiempo
void ComputeFFT(fftwf_complex* signals_t, fftwf_complex* signals_t_reversed, fftwf_complex* signals_f, fftwf_complex* signals_f_reversed,
                int n_signals, int fftsize){

  fftwf_plan plan;
  for(int i=0; i<n_signals; i++){
    plan = fftwf_plan_dft_1d(fftsize, &signals_t[i*fftsize], &signals_f[i*fftsize], FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);

    plan = fftwf_plan_dft_1d(fftsize, &signals_t_reversed[i*fftsize], &signals_f_reversed[i*fftsize], FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(plan);
  }
  fftwf_destroy_plan(plan);
}


//FUNCIÓN PARA CALCULAR EL VALOR ABSOLUTO
  int av(int number){
    return (number<0)? -number:number;
  }


void inverseFFT(fftwf_complex *corr_f, fftwf_complex *corr_t, fftwf_plan plan, int event_length, int shift, int fftsize,
                   float &val_pos, int &lag_pos, float &val_neg, int &lag_neg){
  int tid = omp_get_thread_num();

  fftwf_execute(plan);

  float pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  float av_max = 0.0; //máximo valor absoluto obtenido

  for(int i = event_length-shift; i < event_length+shift; i++){
    if(corr_t[i][0]>pos){
      pos = corr_t[i][0];
      l_pos = i - event_length + 1;
    }
    else if(corr_t[i][0]<neg){
      neg = corr_t[i][0];
      l_neg = i - event_length + 1;
    }
  }

  //av_max = (abs(pos)>abs(neg))? pos:abs(neg);
  val_pos = pos/(float)fftsize;
  lag_pos = l_pos;
  val_neg = neg/(float)fftsize;
  lag_neg = l_neg;
}

void sweep(fftw_complex *xcorrij, float &val_pos, int &lag_pos, float &val_neg, int &lag_neg, int fftsize){
  //en realidad deberían ser de tipo int, al menos los lags, pero no sé si puede haber problemas al pasar las variables a python
  //además, hay que calcular los lags relativos al centro de la función.
  float pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  for(int i=0; i<fftsize; i++){
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

void ComputeNorms(fftwf_complex *events, float *norms,int n_events, int event_length, int fftsize){
  float pot =0.0;
  for(int i=0; i<n_events; i++){
    pot=0.0;
    for(int j=0; j<event_length; j++){
      pot += events[i*fftsize+j][0] * events[i*fftsize+j][0];
    }
    norms[i] = sqrt(pot);
  }
}


//Funcion que implementa la llamada propiamente dicha, haciendo uso del resto de funciones
//arriba declaradas
extern "C"{
  void correlationCPP(fftwf_complex *events, fftwf_complex *events_reversed , int n_events, int event_length,
                      int shift, int fftsize, int num_threads, int chunk_size, float *xcorr_vals_pos, int *xcorr_lags_pos,
                      float *xcorr_vals_neg, int *xcorr_lags_neg){ //añadir las señales de salida
    /*
    events:   conjunto de señales en el tiempo y con zero-appended
    events_reversed: conjunto de señales en el tiempo reversed y zero-appended
    n_events: len(events)    -- numero de series temporales en events
    length_event:            -- tamaño de los datos de una serie temporal
    shift:       int         -- 2*shift+1 será el tamaño de la correlacion resultante (con ind=0 en el punto central)
    */

    printf("C: Going to print the values received from Python:\n");
    printf("C: n_events: %d\n", n_events);
    printf("C: event_length: %d\n", event_length);
    printf("C: shift: %d\n", shift);
    printf("C: fftsize: %d\n\n", fftsize);

    fftwf_complex* signals_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) *(n_events*fftsize));
    fftwf_complex* signals_reversed_freq = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) *(n_events*fftsize));
    printf("C: He reservado memoria satisfactoriamente\n");

    float norms[n_events];
    printf("C: He reservado la memoria de las normas satisfactoriamente\n");

    ComputeNorms(events, norms, n_events, event_length, fftsize);
    printf("C: He computado las normas satisfactoriamente\n");

    ComputeFFT(events, events_reversed, signals_freq, signals_reversed_freq, n_events, fftsize);
    printf("C: He computado las FFT satisfactoriamente\n");

    printf("C: El número de threads es %d\n", num_threads);
    printf("C: El tamaño del chunk es %d\n", chunk_size);
    printf("C: Comienza la extracción de características\n");

    omp_set_num_threads(num_threads);
    #pragma omp parallel default(shared)//private(corr_f, corr_t)
    {
      int tid = omp_get_thread_num();
      fftwf_complex* corr_f = (fftwf_complex*) fftwf_malloc (sizeof(fftwf_complex) *fftsize);
      fftwf_complex* corr_t = (fftwf_complex*) fftwf_malloc (sizeof(fftwf_complex) *fftsize);
      #pragma omp critical
      fftwf_plan plan = fftwf_plan_dft_1d(fftsize, corr_f, corr_t, FFTW_BACKWARD, FFTW_MEASURE);

      #pragma omp for schedule(static)//(dynamic, chunk_size)
      for(int i=0; i<n_events; i++){
        for(int j=i; j<n_events; j++){ //tengo que mirar como hace scipy la correlacion
          ElementWiseMultiplication(&signals_freq[i*fftsize], &signals_reversed_freq[j*fftsize], corr_f, fftsize);

          inverseFFT(corr_f, corr_t, plan, event_length, shift, fftsize, xcorr_vals_pos[i*n_events+j],
                     xcorr_lags_pos[i*n_events+j], xcorr_vals_neg[i*n_events+j], xcorr_lags_neg[i*n_events+j]);

          xcorr_vals_pos[i*n_events+j] /= (norms[i]*norms[j]);
          xcorr_vals_neg[i*n_events+j] /= (norms[i]*norms[j]);
        }
      }
      fftwf_free(corr_f);
      fftwf_free(corr_t);
      fftwf_destroy_plan(plan);
    }

    printf("C: He terminado de computar el bucle\n");

    fftwf_cleanup();
  }
}
