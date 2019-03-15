#include <cstdlib>
#include <iostream>
#include <arrayfire.h>
#include <time.h>
#include <stdio.h>

af::array matrixNorm(af::array x, int axis){
  if(axis == 0){
    return af::sqrt(af::sum(af::pow(x, 2)));
  }
  else{
    return af::sqrt(af::sum(af::pow(x, 1)));
  }
}

extern "C"{

  void toDim4(const unsigned ndims, const dim_t *const dims, af::dim4 d4) {
    af::dim4 d(1, 1, 1, 1);

    for (unsigned i = 0; i < ndims; i++) {
        d[i] = dims[i];
    }
    d4=d;
  }

  void correlationAF(double *events, int n_events, int event_length,
                     int shift, /*int fftsize,*/ double *xcorr_vals_pos, int *xcorr_lags_pos,
                     double *xcorr_vals_neg, int *xcorr_lags_neg){

    af::setBackend(AF_BACKEND_CPU);
    /**
     * events: signals in time with the same length
     * shift : size of the zone inside the temporal correlation that will be checked
     * the rest are output arrays with values and lags, respectively
     **/

    printf("C: Just entered the AF-function\n");

    af::array tss(event_length, n_events, events);

    af::array af_xcorr_lags_pos, af_xcorr_lags_neg, af_xcorr_vals_pos, af_xcorr_vals_neg;


    printf("C: Created the AF-array!\n");
    af_print(tss);
    printf("C: Printed the values received properly\n");

    printf("C: There are %d signals and each one has %d elements\n", n_events, event_length);

    af::array norms = af::matmul(matrixNorm(tss, 0).T(), matrixNorm(tss, 0));
    printf("C: Finished computing the norms\n");

    af::array conv = af::constant(0, tss.dims(1), tss.dims(1), 2*tss.dims(0)-1, tss.type());
    af::array tss_flipped = af::flip(tss, 0);

    af::array prueba = af::convolve(tss(af::span, 1), tss_flipped(af::span, 2, 0),
                                  AF_CONV_EXPAND, AF_CONV_FREQ);
    af_print(prueba);

    for(int row=0; row<n_events; row++){
      printf("Row: %d\n", row);
      //for(int column=row; column<n_events; column++){
      gfor(af::seq column, row, n_events-1){
        conv(row, column, af::span) = af::convolve(tss(af::span, row), tss_flipped(af::span, column
                                      /*af::flip(tss(af::span, column)*/, 0), AF_CONV_EXPAND, AF_CONV_FREQ);
      }
    }
    //af_print(conv);
    printf("C: I have finished the computation\n");

    //af_print(conv.slices(event_length-1-shift, event_length-1+shift));


    af::max(af_xcorr_vals_pos, af_xcorr_lags_pos, conv.slices(event_length - 1 - shift,
                                                  event_length - 1 + shift), 2);
    af::min(af_xcorr_vals_neg, af_xcorr_lags_neg, conv.slices(event_length - 1 - shift,
                                                  event_length - 1 + shift), 2);

    af_xcorr_vals_pos = af_xcorr_vals_pos / norms;
    af_xcorr_vals_neg = af_xcorr_vals_neg / norms;
    af_xcorr_lags_pos = af::upper(af_xcorr_lags_pos.as(af::dtype::s32) - shift);
    af_xcorr_lags_neg = af::upper(af_xcorr_lags_neg.as(af::dtype::s32) - shift);

    af_xcorr_vals_pos.T().host(xcorr_vals_pos);
    af_xcorr_vals_neg.T().host(xcorr_vals_neg);
    af_xcorr_lags_pos.T().host(xcorr_lags_pos);
    af_xcorr_lags_neg.T().host(xcorr_lags_neg);
  }
}

int main(int argc, char * argv[]){

    int backend;
    /*
     AF_BACKEND_DEFAULT = 0,
     AF_BACKEND_CPU     = 1,
     AF_BACKEND_CUDA    = 2,
     AF_BACKEND_OPENCL  = 4,
    */
    if (argc > 1){
        backend = atoi(argv[1]);
        switch(backend){
            case 0: af::setBackend(AF_BACKEND_DEFAULT);
            break;
            case 1: af::setBackend(AF_BACKEND_CPU);
            break;
            case 2: af::setBackend(AF_BACKEND_CUDA);
            break;
            case 3: af::setBackend(AF_BACKEND_OPENCL);
            break;
            default: af::setBackend(AF_BACKEND_DEFAULT);
            break;
        }
    }

	std::cout << "Get available backends: " << af::getAvailableBackends() << std::endl;
	std::cout << "Get active backend: " << af::getActiveBackend() << std::endl;
	std::cout << "Get backend info: " << af::infoString() << std::endl;

  af::array a = af::randn(100, 127);
  //af_print(a);
  //af_print(a(0, af::span));

  int n_events = static_cast<int>(a.dims(0));
  af::array xcorr_vals_pos;
  af::array xcorr_vals_neg;
  af::array xcorr_lags_pos = af::constant(0, n_events, n_events, af::dtype::s32);
  af::array xcorr_lags_neg = af::constant(0, n_events, n_events, af::dtype::s32);

  clock_t start, end;
  double cpu_time_used;
  start = clock();

	//correlationAF(a, 1, xcorr_vals_pos, xcorr_lags_pos, xcorr_vals_neg, xcorr_lags_neg);

  end = clock();
  printf("CPU USED TIME: %f\n", ((double)(end-start))/CLOCKS_PER_SEC);

	return 0;
}
