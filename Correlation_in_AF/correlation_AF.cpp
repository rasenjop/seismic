#include <cstdlib>
#include <iostream>
#include <arrayfire.h>
#include <time.h>
#include <stdio.h>
#include <chrono>

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

  void correlationAF(float *events, int n_events, int event_length,
                     int shift, /*int fftsize,*/ float *xcorr_vals_pos, int *xcorr_lags_pos,
                     float *xcorr_vals_neg, int *xcorr_lags_neg){
    af::setBackend(AF_BACKEND_DEFAULT);
    std::cout << "Get available backends: " << af::getAvailableBackends() << std::endl;
    std::cout << "Get backend counts: " << af::getBackendCount() << std::endl;
    std::cout << "Get active backend: " << af::getActiveBackend() << std::endl;
    std::cout << "Get backend info: " << af::infoString() << std::endl;


    printf("C: Just entered the AF-function\n");

    af::array tss(event_length, n_events, events);

    af::array af_xcorr_vals_pos = af::constant(0, tss.dims(1), tss.dims(1), af::dtype::f32);
    af::array af_xcorr_vals_neg = af::constant(0, tss.dims(1), tss.dims(1), af::dtype::f32);
    af::array af_xcorr_lags_pos = af::constant(0, tss.dims(1), tss.dims(1), af::dtype::s32);
    af::array af_xcorr_lags_neg = af::constant(0, tss.dims(1), tss.dims(1), af::dtype::s32);

    printf("C: Created the AF-array!\n");
    //af_print(tss);
    printf("C: Printed the values received properly\n");

    printf("C: There are %d signals and each one has %d elements\n", n_events, event_length);

    af::array norms = af::matmul(matrixNorm(tss, 0).T(), matrixNorm(tss, 0));
    printf("C: Finished computing the norms\n");

    af::array conv = af::constant(0, tss.dims(0), tss.dims(1), tss.type());
    af::array tss_flipped = af::flip(tss, 0);

    af::array index;
    af::array val;
    af::array k;
    af::array kk;

    for(int row=0; row<n_events; row++){
      //for(int column=row; column<n_events; column++){
      conv = af::constant(0, tss.dims(0), tss.dims(1), tss.type());

      gfor(af::seq column, row, n_events-1){
        conv(af::span, column) = af::convolve(tss(af::span, row), tss_flipped(af::span, column
                                    /*af::flip(tss(af::span, column)*/, 0), AF_CONV_DEFAULT, AF_CONV_FREQ);
      }

      af::max(k, kk, conv.rows(event_length/2 - shift, event_length/2 + shift), 0);
      af_xcorr_vals_pos(row, af::span) = k;
      af_xcorr_lags_pos(row, af::span) = kk;
      af::min(k, kk, conv.rows(event_length/2 - shift, event_length/2 + shift), 0);
      af_xcorr_vals_neg(row, af::span) = k;
      af_xcorr_lags_neg(row, af::span) = kk;
    }

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
