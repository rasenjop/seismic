#include "correlation_c.h"


#include <vector>
#include <fftw3.h>
#include <cmath>
#include <thread>
#include <mutex>

using namespace std; //es necesario? ¿Por qué?

/*Tags empleados:
- TODO: por implementar o cambios a realizar
- PROBLEMA
- DUDA
*/
mutex mtx;


double normalize(fftw_complex* signal1_t, fftw_complex* signal2_t, int event_length){ //normalizes a pair of signals
  double pot_1 = 0.0, pot_2 = 0.0;

  for(int i=0; i<event_length; i++){
    pot_1 += signal1_t[i][0]*signal1_t[i][0];
    pot_2 += signal2_t[i][0]*signal2_t[i][0];
  }
  return sqrt(pot_1*pot_2);
}



//aqui la señal llega ya recortada (250 muestras) y en el dominio del tiempo
void ComputeFFT(fftw_complex* signals_t, fftw_complex* signals_t_reversed, fftw_complex* signals_f, fftw_complex* signals_f_reversed,
                int n_signals, int fftsize){

  //si se conoce el tamaño de cada señal, habria que introducir un nuevo parametro y emplearlo
  //en el fftw_plan --> fftsize
  fftw_plan plan;
  //#pragma omp parallel for
  for(int i=0; i<n_signals; i++){
    plan = fftw_plan_dft_1d(fftsize, &signals_t[i*fftsize], &signals_f[i*fftsize], FFTW_FORWARD, FFTW_ESTIMATE);
    //fftw_print_plan(plan);
    fftw_execute(plan);

    //AQUÍ TAMBIÉN HAY QUE HACER LA TRANSFORMADA DE LAS INVERSAS
    plan = fftw_plan_dft_1d(fftsize, &signals_t_reversed[i*fftsize], &signals_f_reversed[i*fftsize], FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    //printf("\n");
    //Para hacer las pruebas ponemos FFTW_ESTIMATE, para hacer el definitivo probaremos con FFTW_MEASURE
  }
  fftw_destroy_plan(plan);
}


//FUNCIÓN PARA CALCULAR EL VALOR ABSOLUTO
  int av(int number){
    return (number<0)? -number:number;
  }


//No se si el FFTW_MEASURE es valido si se hacen llamadas en distintas funciones.
//Es decir, si se crea y se destruye el plan continuamente.
//Quiza seria mejor hacer todas las FFT dentro de un mismo bucle, empleando
//siempre el mismo fftw_plan. Sin embargo, esto no es posible, porque en el plan
//van indicados tanto el vector que contiene la señal en el tiempo como el de la
//señal en frecuencia.

void inverseFFT(fftw_complex *corr_f, int event_length, int shift, int fftsize, double &val_pos,
                                  int &lag_pos, double &val_neg, int &lag_neg){

  fftw_plan plan;
  fftw_complex* corr_t = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *fftsize);

  mtx.lock();
  plan = fftw_plan_dft_1d(fftsize, corr_f, corr_t, FFTW_BACKWARD, FFTW_ESTIMATE);
  mtx.unlock();
  fftw_execute(plan);
  //FFTW_ESTIMATE OR FFTW_MEASURE no sé si al hacerlo todo en llamadas por separado
  //sería conveniente emplear uno u otro
  /*for(int i=0; i<fftsize; i++){
    printf("Parte real tiempo señal %d: %f\n", i, corr_t[i][0]);
    printf("Parte imag tiempo señal %d: %f\n", i, corr_t[i][1]);
  }
  getchar();*/

  double pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  double av_max = 0.0; //máximo valor absoluto obtenido

  // creo que el escalado no nos hace falta, porque las señales de entrada ya están
  // normalizadas y demeaned
  for(int i = event_length-shift; i < event_length+shift; i++){  //PROBLEMA: empezamos el barrido en centro-shift y acabamos en centro+shift?
    //corr_t[i][0] = corr_t[i][0] / (double)fftsize;
    //ESTE ESCALADO PUEDE SER PROBLEMÁTICO -> PUEDE NO OBTENERSE LOS MISMOS
    //VALORES QUE SE OBTIENEN CON correlate.py

    //TODO: HAY QUE INCLUIR LA OBTENCIÓN DE LAS CARACTERÍSTICAS AQUÍ DENTRO,
    //APROVECHANDO EL MISMO BUCLE
    if(corr_t[i][0]>pos){
      pos = corr_t[i][0];
      l_pos = i - event_length + 1;
    }
    else if(corr_t[i][0]<neg){
      neg = corr_t[i][0];
      l_neg = i - event_length + 1;
    }
  }

  /*printf("pos: %f\n", pos);
  printf("neg: %f\n", neg);*/

  //Como finalmente hay que dividir entre el máximo valor absoluto, tiene
  //que ser bien el máximo más positivo o el mínimo más negativo
  //PROBLEMA: Al dividir entre el máximo, uno de los valores siempre será 1,
  //por lo que no obtendremos lo que deseamos
  av_max = (abs(pos)>abs(neg))? pos:abs(neg);
  //TODO: dividir entre la norma y el número de puntos de la señal en frecuencia
  val_pos = pos/(double)fftsize;
  lag_pos = l_pos;
  val_neg = neg/(double)fftsize;
  lag_neg = l_neg;

  fftw_destroy_plan(plan);
  fftw_free(corr_t);
}


void ElementWiseMultiplication(fftw_complex *signal_a, fftw_complex *signal_b, fftw_complex *result, int fftsize){
  for(int i=0; i<fftsize; i++){
    result[i][0] = signal_a[i][0] * signal_b[i][0] - signal_a[i][1] * signal_b[i][1];
    result[i][1] = signal_a[i][1] * signal_b[i][0] + signal_a[i][0] * signal_b[i][1];
  }
}

void ComputeNorms(fftw_complex *events, double *norms,int n_events, int event_length, int fftsize){
  double pot =0.0;
  for(int i=0; i<n_events; i++){
    pot=0.0;
    for(int j=0; j<event_length; j++){
      pot += events[i*fftsize+j][0] * events[i*fftsize+j][0];
    }
    norms[i] = sqrt(pot);
  }
}

void ComputeSegments(int n_events, int num_threads, int *start_point){
  start_point[num_threads] = n_events;
  start_point[num_threads - 1] = n_events - n_events / sqrt(num_threads);
  long const x = n_events * n_events / num_threads;
  int previous = n_events / sqrt(num_threads);
  int diff = 0;
  for(int i = num_threads - 2; i >= 0; i--){
    diff = - previous + sqrt(pow(previous, 2.0) + x);
    start_point[i] = start_point[i + 1] - diff;
    previous += diff;
  }
}


void ThreadComputation(int th_ID, int *start_point, fftw_complex *signals_freq, fftw_complex *signals_reversed_freq,
             int n_events, int event_length, int shift, int fftsize, double *norms, double *xcorr_vals_pos,
             int *xcorr_lags_pos, double *xcorr_vals_neg, int *xcorr_lags_neg){

  fftw_complex* xcorrij_f = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftsize);
  for(int i=start_point[th_ID]; i<start_point[th_ID+1]; i++){
    //printf("I am the thread %d and this iteration is the %d\n", th_ID, i);
    for(int j=i; j<n_events; j++){

      ElementWiseMultiplication(&signals_freq[i*fftsize], &signals_reversed_freq[j*fftsize], xcorrij_f, fftsize);

      inverseFFT(xcorrij_f, event_length, shift, fftsize, xcorr_vals_pos[i*n_events+j], xcorr_lags_pos[i*n_events+j],
                                       xcorr_vals_neg[i*n_events+j], xcorr_lags_neg[i*n_events+j]);
      xcorr_vals_pos[i*n_events+j] /= (norms[i]*norms[j]);
      xcorr_vals_neg[i*n_events+j] /= (norms[i]*norms[j]);
    }
  }
  fftw_free(xcorrij_f);
}


//Funcion que implementa la llamada propiamente dicha, haciendo uso del resto de funciones
//arriba declaradas
extern "C"{
  void correlationTH(fftw_complex *events, fftw_complex *events_reversed , int n_events, int event_length,
                      int shift, int fftsize, int num_threads, double *xcorr_vals_pos, int *xcorr_lags_pos,
                      double *xcorr_vals_neg, int *xcorr_lags_neg){ //añadir las señales de salida
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

    fftw_complex* signals_freq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *(n_events*fftsize));
    fftw_complex* signals_reversed_freq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *(n_events*fftsize));
    printf("C: He reservado memoria satisfactoriamente\n");

    double *norms = new double[n_events];
    printf("C: He reservado la memoria de las normas satisfactoriamente\n");

    ComputeNorms(events, norms, n_events, event_length, fftsize);
    printf("C: He computado las normas satisfactoriamente\n");

    ComputeFFT(events, events_reversed, signals_freq, signals_reversed_freq, n_events, fftsize);
    printf("C: He computado las FFT satisfactoriamente\n");

    int *start_point = new int[num_threads + 1];
    ComputeSegments(n_events, num_threads, start_point);
    printf("C: los puntos iniciales de cada thread son: \n");
    for(int i=0; i<num_threads; i++){
      printf("C: El thread %d empieza en la iteración %d\n", i, start_point[i]);
    }

    vector<thread> threads;

    printf("C: Comienza la extracción de características\n");
    int i=0;


    for (i=0; i<num_threads; i++) {
     	threads.push_back(thread(ThreadComputation, i, start_point, signals_freq, signals_reversed_freq,
                  n_events, event_length, shift, fftsize, norms, xcorr_vals_pos, xcorr_lags_pos,
                  xcorr_vals_neg, xcorr_lags_neg));
    }
    for (auto &t: threads)
      t.join();

    fftw_free(signals_freq);
    fftw_free(signals_reversed_freq);
    delete[] norms;
    delete[] start_point;



    // for(int i=0; i<n_events; i++){
    //   //printf("Iteration %d of %d\n", i, n_events);
    //   for(int j=i; j<n_events; j++){ //tengo que mirar como hace scipy la correlacion
    //
    //     ElementWiseMultiplication(&signals_freq[i*fftsize], &signals_reversed_freq[j*fftsize], xcorrij_f, fftsize);
    //
    //     //IMPLEMENTAR LA FFT INVERSA QUE SE APLICA DE UNO EN UNO --> MEMORY BOUND
    //     inverseFFT(xcorrij_f, event_length, shift, fftsize, xcorr_vals_pos[i*n_events+j], xcorr_lags_pos[i*n_events+j],
    //                                      xcorr_vals_neg[i*n_events+j], xcorr_lags_neg[i*n_events+j]);
    //     //sweep(xcorrij_t, xcorr_vals_pos[i*n_events+j], xcorr_lags_pos[i*n_events+j], xcorr_vals_neg[i*n_events+j], xcorr_lags_neg[i*n_events+j], fftsize);
    //     xcorr_vals_pos[i*n_events+j] /= (norms[i]*norms[j]);
    //     xcorr_vals_neg[i*n_events+j] /= (norms[i]*norms[j]);
    //   }
    // }

    fftw_cleanup();
  }
}
