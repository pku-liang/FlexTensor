#include <ap_int.h>
  
#include <algorithm>

extern "C" void default_function_kernel0( float* C,  float* A,  float* B) {
#pragma HLS INTERFACE m_axi port=C  offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE m_axi port=A  offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE m_axi port=B  offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

  for (int i_inner_outer = 0; i_inner_outer < 32; ++i_inner_outer) {
    for (int j_outer = 0; j_outer < 32; ++j_outer) {
      for (int i_inner_inner_init = 0; i_inner_inner_init < 32; ++i_inner_inner_init) {
        for (int j_inner_init = 0; j_inner_init < 16; ++j_inner_init) {
          C[((((i_inner_outer * 16384) + (i_inner_inner_init * 512)) + (j_outer * 16)) + j_inner_init)] = 0.000000e+00f;
        }
      }
      for (int k_outer = 0; k_outer < 32; ++k_outer) {
        for (int i_inner_inner = 0; i_inner_inner < 32; ++i_inner_inner) {
          for (int k_inner = 0; k_inner < 8; ++k_inner) {
            for (int j_inner = 0; j_inner < 16; ++j_inner) {
              C[((((i_inner_outer * 16384) + (i_inner_inner * 512)) + (j_outer * 16)) + j_inner)] = (C[((((i_inner_outer * 16384) + (i_inner_inner * 512)) + (j_outer * 16)) + j_inner)] + (A[((((i_inner_outer * 8192) + (i_inner_inner * 256)) + (k_outer * 8)) + k_inner)] * B[((((k_outer * 4096) + (k_inner * 512)) + (j_outer * 16)) + j_inner)]));
            }
          }
        }
      }
    }
  }
}
