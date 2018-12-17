#include "correlation_c.h"


#include <vector>
#include <fftw3.h>
#include <cmath>

using namespace std; //es necesario? ¿Por qué?

/*Tags empleados:
- TODO: por implementar o cambios a realizar
- PROBLEMA
- DUDA
*/


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

void inverseFFT(fftw_complex *corr_f, int fftsize, int shift, double &val_pos,
                                  int &lag_pos, double &val_neg, int &lag_neg){

  fftw_plan plan;
  fftw_complex* corr_t = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) *fftsize);

  plan = fftw_plan_dft_1d(fftsize, corr_f, corr_t, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(plan);
  //FFTW_ESTIMATE OR FFTW_MEASURE no sé si al hacerlo todo en llamadas por separado
  //sería conveniente emplear uno u otro

  double pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  double av_max = 0.0; //máximo valor absoluto obtenido

  // creo que el escalado no nos hace falta, porque las señales de entrada ya están
  // normalizadas y demeaned
  for(int i = fftsize/2-shift; i <= fftsize/2+shift; i++){  //PROBLEMA: empezamos el barrido en centro-shift y acabamos en centro+shift?
    //corr_t[i][0] = corr_t[i][0] / (double)fftsize;
    //ESTE ESCALADO PUEDE SER PROBLEMÁTICO -> PUEDE NO OBTENERSE LOS MISMOS
    //VALORES QUE SE OBTIENEN CON correlate.py

    //TODO: HAY QUE INCLUIR LA OBTENCIÓN DE LAS CARACTERÍSTICAS AQUÍ DENTRO,
    //APROVECHANDO EL MISMO BUCLE
    if(corr_t[i][0]> pos){
      pos = corr_t[i][0];
      l_pos = i;
    }
    else if(corr_t[i][0]<neg){
      neg = corr_t[i][0];
      l_neg = i;
    }
  }

  //Como finalmente hay que dividir entre el máximo valor absoluto, tiene
  //que ser bien el máximo más positivo o el mínimo más negativo
  //PROBLEMA: Al dividir entre el máximo, uno de los valores siempre será 1,
  //por lo que no obtendremos lo que deseamos
  av_max = (abs(pos)>abs(neg))? pos:abs(neg);
  //TODO: dividir entre la norma y el número de puntos de la señal en frecuencia
  val_pos = pos/(double)fftsize;
  lag_pos = l_pos-shift;
  val_neg = neg/(double)fftsize;
  lag_neg = l_neg-shift;

  fftw_destroy_plan(plan);
  fftw_free(corr_t);
}

void sweep(fftw_complex *xcorrij, double &val_pos, int &lag_pos, double &val_neg, int &lag_neg, int fftsize){
  //en realidad deberían ser de tipo int, al menos los lags, pero no sé si puede haber problemas al pasar las variables a python
  //además, hay que calcular los lags relativos al centro de la función.
  double pos=0.0, neg=0.0;
  int l_pos=0, l_neg=0;

  for(int i=0; i<fftsize; i++){
    if(xcorrij[i][0]> pos){
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
void reverseArray(fftw_complex *signal, int size){
  fftw_complex aux;
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
  fftw_free(aux);
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


//Funcion que implementa la llamada propiamente dicha, haciendo uso del resto de funciones
//arriba declaradas
extern "C"{
  void correlationCPP(fftw_complex *events, fftw_complex *events_reversed , int n_events, int event_length,
                      int shift, int fftsize, double *xcorr_vals_pos, int *xcorr_lags_pos,
                      double *xcorr_vals_neg, int *xcorr_lags_neg){ //añadir las señales de salida
    /*
    events:   conjunto de señales en el tiempo y con zero-appended
    events_reversed: conjunto de señales en el tiempo reversed y zero-appended
    n_events: len(events)    -- numero de series temporales en events
    length_event:            -- tamaño de los datos de una serie temporal
    shift:       int         -- 2*shift+1 será el tamaño de la correlacion resultante (con ind=0 en el punto central)
    */

    //int fftsize = 2*n_events+1;
    fftw_complex signals_freq[n_events*fftsize];
    fftw_complex signals_reversed_freq[n_events*fftsize];
    //En los resultados obtenidos con correlate
    //Siempre se obtiene un tamaño de 2*shift+1. He de investigar pq

    double norms[n_events];

    ComputeNorms(events, norms, n_events, event_length, fftsize);

    ComputeFFT(events, events_reversed, signals_freq, signals_reversed_freq, n_events, fftsize);

    fftw_complex* xcorrij_f = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*fftsize);

    /*for(int i=0; i<fftsize; i++){
      printf("Parte real %d: %f\n", i,events[i][0]);
      printf("Parte imaginaria %d: %f\n", i,events[i][1]);
    }*/


    for(int i=0; i<n_events; i++){
      //printf("Iteration %d of %d\n", i, n_events);
      for(int j=i; j<n_events; j++){ //tengo que mirar como hace scipy la correlacion

        ElementWiseMultiplication(&signals_freq[i*fftsize], &signals_reversed_freq[j*fftsize], xcorrij_f, fftsize);

        //IMPLEMENTAR LA FFT INVERSA QUE SE APLICA DE UNO EN UNO --> MEMORY BOUND
        inverseFFT(xcorrij_f, fftsize, shift, xcorr_vals_pos[i*n_events+j], xcorr_lags_pos[i*n_events+j],
                                         xcorr_vals_neg[i*n_events+j], xcorr_lags_neg[i*n_events+j]);
        //sweep(xcorrij_t, xcorr_vals_pos[i*n_events+j], xcorr_lags_pos[i*n_events+j], xcorr_vals_neg[i*n_events+j], xcorr_lags_neg[i*n_events+j], fftsize);
        xcorr_vals_pos[i*n_events+j] /= (norms[i]*norms[j]);
        xcorr_vals_neg[i*n_events+j] /= (norms[i]*norms[j]);
      }
    }
    fftw_cleanup();
    fftw_free(xcorrij_f);
  }
}




/**************************************************************************************************/
/*int main(){
  int event_length = 10;
  int fftsize = 2*event_length+1;
  fftw_complex signal1_t[fftsize]; //relleno de ceros al final, hasta fftsize
  fftw_complex signal2_t[fftsize]; //ya invertido, con zeros al final

  fftw_complex signal1_f[fftsize];
  fftw_complex signal2_f[fftsize];

  for(int i=0; i<fftsize; i++){
    if(i<event_length){
      signal1_t[i][0] =-0.9 + (i*0.1);
      signal2_t[i][0] = 0.6 - (i*0.1);
    }
    else{
      signal1_t[i][0] = 0.0;
      signal2_t[i][0] = 0.0;
    }

    signal1_t[i][1] =0.0;
    signal2_t[i][1] =0.0;

    signal1_f[i][0] = 0.0;
    signal2_f[i][0] = 0.0;

    signal1_f[i][1] = 0.0;
    signal2_f[i][1] = 0.0;
  }

  fftw_plan plan;

  plan = fftw_plan_dft_1d(fftsize, signal1_t, signal1_f, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_print_plan(plan);
  printf("\n");
  //Para hacer las pruebas ponemos FFTW_ESTIMATE, para hacer el definitivo probaremos con FFTW_MEASURE
  fftw_execute(plan);

  plan = fftw_plan_dft_1d(fftsize, signal2_t, signal2_f, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_print_plan(plan);
  printf("\n");

  fftw_execute(plan);

  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i,signal1_f[i][0]);
  }
  printf("LA SEGUNDA FFT:\n");
  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i,signal2_f[i][0]);
  }

  //Ahora calculamos la multipllicación elemento a elemento y hallaremos la ifft

  fftw_complex xcorrij_f[fftsize];
  fftw_complex xcorrij_t[fftsize];

  ElementWiseMultiplication(signal1_f, signal2_f, xcorrij_f, fftsize);

  printf("LA SEÑAL DE CORRELACION EN LA FRECUENCIA:\n");
  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i,xcorrij_f[i][0]);
  }

  plan = fftw_plan_dft_1d(fftsize, xcorrij_f, xcorrij_t, FFTW_BACKWARD, FFTW_ESTIMATE);
  fftw_execute(plan);

  double norm = normalize(signal1_t, signal2_t, event_length);
  //multiplicamos por fftsize para hacer la normalización completa de la IFFT -> viene en la ecuación
  norm *= fftsize;

  printf("LA SEÑAL DE CORRELACION EN EL TIEMPO:\n");
  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i,xcorrij_t[i][0]/norm);
  }



  return 0;
}*/
  /*int temp_size = 10; //tamaño de cada señal temporal (ya con el zero-appending)
  int fftsize = 2*temp_size+1; //tamaño de la fft resultante
  fftw_complex signals_t[fftsize]; //array linearizado, en este caso con sólo dos señales
  fftw_complex signals_f[fftsize];    //array linearizado, con dos señales, de las ffts
  //fftw_complex signal2_t[tam];
  for(int i=0; i<fftsize; i++){
    if(i<temp_size){
      signals_t[i][0] = float(i);
      signals_t[i][1] = 0.0;
    }
    else{
      signals_t[i][0] = 0.0;
      signals_t[i][1] = 0.0;
    }
  }

  for(int i=0; i<fftsize; i++){
    signals_f[i][0] = 0.0;
    signals_f[i][1] = 0.0;
  }

  //fftw_complex *signal2_t = &signals_t[temp_size];
  ComputeFFT(signals_t, signals_f, 1, fftsize); //se está produciendo un abort trap 6 aquí dentro

  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i,signals_f[i][0]);
  }
*/
  /**********************************************************/

  /*fftw_complex signal1_t[fftsize];
  fftw_complex signal1_f[fftsize];


  for(int i=0; i<fftsize; i++){
    if(i<temp_size){
      signal1_t[i][0] = temp_size*2-1 - float(i);
      signal1_t[i][1] = 0.0;
    }
    else{
      signal1_t[i][0] = 0.0;
      signal1_t[i][1] = 0.0;
    }
  }
  printf("SE IMPRIME LA CADENA INVERTIDA:\n");
  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i, signal1_t[i][0]);
  }

  for(int i=0; i<fftsize; i++){
    signal1_f[i][0] = 0.0;
    signal1_f[i][1] = 0.0;
  }

  fftw_plan plan;

  plan = fftw_plan_dft_1d(fftsize, signal1_t, signal1_f, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_print_plan(plan);
  printf("\n");
  //Para hacer las pruebas ponemos FFTW_ESTIMATE, para hacer el definitivo probaremos con FFTW_MEASURE
  fftw_execute(plan);
  fftw_destroy_plan(plan);

  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i, signal1_f[i][0]);
  }


*/
  /**********************************************************/

  /*fftw_complex xcorrij[fftsize];
  fftw_complex xcorrij_t[fftsize];

  ElementWiseMultiplication(signals_f, signal1_f, xcorrij, fftsize);

  //IMPLEMENTAR LA FFT INVERSA QUE SE APLICA DE UNO EN UNO --> MEMORY BOUND
  inverseFFT(xcorrij, xcorrij_t, fftsize);

  printf("SE IMPRIME AHORA LA FUNCIÓN DE CORRELACIÓN:\n");

  for(int i=0; i<fftsize; i++){
    printf("El valor %i es: %f\n", i, xcorrij_t[i][0]);
  }

  return 0;
}*/
