#include <cstdlib>
#include <iostream>
#include <arrayfire.h>
#include <time.h>

af::array matrixNorm(af::array x, int axis){
  if(axis == 0){
    return af::sqrt(af::sum(af::pow(x, 2)));
  }
  else{
    return af::sqrt(af::sum(af::pow(x, 1)));
  }
}

extern "C"{

  void correlationAF(af::array tss, int shift, af::array &xcorr_vals_pos,
                    af::array &xcorr_lags_pos, af::array &xcorr_vals_neg,
                    af::array &xcorr_lags_neg){
    /**
     * events: signals in time with the same length
     * shift : size of the zone inside the temporal correlation that will be checked
     * the rest are output arrays with values and lags, respectively
     **/

    printf("C: Just entered the AF-function\n");
    int length_event = static_cast<int>(tss.dims(0));
    int n_events = static_cast<int>(tss.dims(1));
    printf("C: There are %d signals and each one has %d elements\n", n_events, length_event);

    af::array norms = af::matmul(matrixNorm(tss, 0).T(), matrixNorm(tss, 0));
    //af_print(norms);

    af::array conv = af::constant(0, tss.dims(1), tss.dims(1), 2*tss.dims(0)-1, tss.type());
    af::array tss_flipped = af::flip(tss, 0);

    for(int row=0; row<n_events; row++){
      printf("Row: %d\n", row);
      for(int column=row; column<n_events; column++){
      /*gfor(af::seq column, row, n_events-1){*/
        conv(row, column, af::span) = af::convolve(tss(af::span, row), tss_flipped(af::span, column
                                      /*af::flip(tss(af::span, column)*/, 0), AF_CONV_EXPAND, AF_CONV_FREQ);
      }
    }
    //af_print(conv);
    printf("I have finished the computation\n");

    af::max(xcorr_vals_pos, xcorr_lags_pos, conv, 2);
    af::min(xcorr_vals_neg, xcorr_lags_neg, conv, 2);
    xcorr_vals_pos = xcorr_vals_pos / norms;
    xcorr_vals_neg = xcorr_vals_neg / norms;
    xcorr_lags_pos = af::upper(xcorr_lags_pos.as(af::dtype::s32)  - length_event + 1);
    xcorr_lags_neg = af::upper(xcorr_lags_neg.as(af::dtype::s32)  - length_event + 1);
    //TODO : incluir el shift


    /*af_print(xcorr_vals_pos);
    af_print(xcorr_vals_neg);
    af_print(xcorr_lags_pos);
    af_print(xcorr_lags_neg);*/

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

	correlationAF(a, 1, xcorr_vals_pos, xcorr_lags_pos, xcorr_vals_neg, xcorr_lags_neg);

  end = clock();
  printf("CPU USED TIME: %f\n", ((double)(end-start))/CLOCKS_PER_SEC);

	return 0;
}
