/* 
    yolo conv layer 11 shape:
    batch = 8
    input channel = 512
    height = 14
    width = 14
    output channel = 1024
    kernel size = 3x3
    padding = 1
    stride = 1
*/
__kernel void default_function_kernel0(__global float* restrict data, __global float* restrict kernel, __global float* restrict compute) {
   float compute_local[16];
  __local float pad_temp_shared[112];
  __local float kernel_shared[512];
  compute_local[0] = 0.000000e+00f;
  compute_local[8] = 0.000000e+00f;
  compute_local[4] = 0.000000e+00f;
  compute_local[12] = 0.000000e+00f;
  compute_local[1] = 0.000000e+00f;
  compute_local[9] = 0.000000e+00f;
  compute_local[5] = 0.000000e+00f;
  compute_local[13] = 0.000000e+00f;
  compute_local[2] = 0.000000e+00f;
  compute_local[10] = 0.000000e+00f;
  compute_local[6] = 0.000000e+00f;
  compute_local[14] = 0.000000e+00f;
  compute_local[3] = 0.000000e+00f;
  compute_local[11] = 0.000000e+00f;
  compute_local[7] = 0.000000e+00f;
  compute_local[15] = 0.000000e+00f;
  for (int rc_outer = 0; rc_outer < 256; ++rc_outer) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if ((((int)get_local_id(2)) * 7) < (112 - ((int)get_local_id(0)))) {
      if (((int)get_local_id(0)) < 7) {
        if ((((int)get_group_id(2)) / 4) < (8 - (((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 112))) {
          pad_temp_shared[((((((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 112) * 112) + ((((((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 14) % 2) * 14) + (((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) % 14))) + (((((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 28) % 4) * 28))] = data[((((((((((int)get_group_id(2)) / 4) * 7168) + ((int)get_group_id(1))) + (rc_outer * 28)) + ((((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 112) * 7168)) * 28) + ((((((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 14) % 2) * 14) + (((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) % 14))) + (((((((int)get_local_id(2)) * 7) + ((int)get_local_id(0))) / 28) % 4) * 196))];
        }
      }
    }
    if ((((int)get_local_id(2)) * 8) < (128 - ((((int)get_local_id(0)) * 3) / 4))) {
      if ((((int)get_local_id(2)) * 32) < (512 - (((int)get_local_id(0)) * 3))) {
        if (((int)get_local_id(0)) < 11) {
          if (((((int)get_group_id(2)) % 4) * 128) < (512 - ((((int)get_local_id(2)) * 8) + ((((int)get_local_id(0)) * 3) / 4)))) {
            kernel_shared[((((((int)get_local_id(2)) * 8) + ((((int)get_local_id(0)) * 3) / 4)) * 4) + ((((int)get_local_id(0)) * 3) % 4))] = kernel[(((((((((int)get_group_id(2)) % 4) * 32768) + rc_outer) + (((int)get_local_id(2)) * 2048)) + (((((int)get_local_id(0)) * 3) / 4) * 256)) * 4) + ((((int)get_local_id(0)) * 3) % 4))];
          }
        }
      }
    }
    if ((((int)get_local_id(2)) * 8) < (128 - (((((int)get_local_id(0)) * 3) + 1) / 4))) {
      if ((((int)get_local_id(2)) * 32) < (511 - (((int)get_local_id(0)) * 3))) {
        if (((int)get_local_id(0)) < 11) {
          if (((((int)get_group_id(2)) % 4) * 128) < (512 - ((((int)get_local_id(2)) * 8) + (((((int)get_local_id(0)) * 3) + 1) / 4)))) {
            kernel_shared[((((((int)get_local_id(2)) * 8) + (((((int)get_local_id(0)) * 3) + 1) / 4)) * 4) + (((((int)get_local_id(0)) * 3) + 1) % 4))] = kernel[(((((((((int)get_group_id(2)) % 4) * 32768) + rc_outer) + (((int)get_local_id(2)) * 2048)) + ((((((int)get_local_id(0)) * 3) + 1) / 4) * 256)) * 4) + (((((int)get_local_id(0)) * 3) + 1) % 4))];
          }
        }
      }
    }
    if ((((int)get_local_id(2)) * 8) < (128 - (((((int)get_local_id(0)) * 3) + 2) / 4))) {
      if ((((int)get_local_id(2)) * 32) < (510 - (((int)get_local_id(0)) * 3))) {
        if (((int)get_local_id(0)) < 10) {
          if (((((int)get_group_id(2)) % 4) * 128) < (512 - ((((int)get_local_id(2)) * 8) + (((((int)get_local_id(0)) * 3) + 2) / 4)))) {
            kernel_shared[((((((int)get_local_id(2)) * 8) + (((((int)get_local_id(0)) * 3) + 2) / 4)) * 4) + (((((int)get_local_id(0)) * 3) + 2) % 4))] = kernel[(((((((((int)get_group_id(2)) % 4) * 32768) + rc_outer) + (((int)get_local_id(2)) * 2048)) + ((((((int)get_local_id(0)) * 3) + 2) / 4) * 256)) * 4) + (((((int)get_local_id(0)) * 3) + 2) % 4))];
          }
        }
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    compute_local[0] = (compute_local[0] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[(((int)get_local_id(2)) * 16)]));
    compute_local[8] = (compute_local[8] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 256)]));
    compute_local[4] = (compute_local[4] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[(((int)get_local_id(2)) * 16)]));
    compute_local[12] = (compute_local[12] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 256)]));
    compute_local[1] = (compute_local[1] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 4)]));
    compute_local[9] = (compute_local[9] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 260)]));
    compute_local[5] = (compute_local[5] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 4)]));
    compute_local[13] = (compute_local[13] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 260)]));
    compute_local[2] = (compute_local[2] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 8)]));
    compute_local[10] = (compute_local[10] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 264)]));
    compute_local[6] = (compute_local[6] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 8)]));
    compute_local[14] = (compute_local[14] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 264)]));
    compute_local[3] = (compute_local[3] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 12)]));
    compute_local[11] = (compute_local[11] + (pad_temp_shared[((int)get_local_id(0))] * kernel_shared[((((int)get_local_id(2)) * 16) + 268)]));
    compute_local[7] = (compute_local[7] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 12)]));
    compute_local[15] = (compute_local[15] + (pad_temp_shared[(((int)get_local_id(0)) + 14)] * kernel_shared[((((int)get_local_id(2)) * 16) + 268)]));
    compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 1)]));
    compute_local[8] = (compute_local[8] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 257)]));
    compute_local[4] = (compute_local[4] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 1)]));
    compute_local[12] = (compute_local[12] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 257)]));
    compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 5)]));
    compute_local[9] = (compute_local[9] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 261)]));
    compute_local[5] = (compute_local[5] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 5)]));
    compute_local[13] = (compute_local[13] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 261)]));
    compute_local[2] = (compute_local[2] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 9)]));
    compute_local[10] = (compute_local[10] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 265)]));
    compute_local[6] = (compute_local[6] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 9)]));
    compute_local[14] = (compute_local[14] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 265)]));
    compute_local[3] = (compute_local[3] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 13)]));
    compute_local[11] = (compute_local[11] + (pad_temp_shared[(((int)get_local_id(0)) + 28)] * kernel_shared[((((int)get_local_id(2)) * 16) + 269)]));
    compute_local[7] = (compute_local[7] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 13)]));
    compute_local[15] = (compute_local[15] + (pad_temp_shared[(((int)get_local_id(0)) + 42)] * kernel_shared[((((int)get_local_id(2)) * 16) + 269)]));
    compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 2)]));
    compute_local[8] = (compute_local[8] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 258)]));
    compute_local[4] = (compute_local[4] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 2)]));
    compute_local[12] = (compute_local[12] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 258)]));
    compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 6)]));
    compute_local[9] = (compute_local[9] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 262)]));
    compute_local[5] = (compute_local[5] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 6)]));
    compute_local[13] = (compute_local[13] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 262)]));
    compute_local[2] = (compute_local[2] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 10)]));
    compute_local[10] = (compute_local[10] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 266)]));
    compute_local[6] = (compute_local[6] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 10)]));
    compute_local[14] = (compute_local[14] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 266)]));
    compute_local[3] = (compute_local[3] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 14)]));
    compute_local[11] = (compute_local[11] + (pad_temp_shared[(((int)get_local_id(0)) + 56)] * kernel_shared[((((int)get_local_id(2)) * 16) + 270)]));
    compute_local[7] = (compute_local[7] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 14)]));
    compute_local[15] = (compute_local[15] + (pad_temp_shared[(((int)get_local_id(0)) + 70)] * kernel_shared[((((int)get_local_id(2)) * 16) + 270)]));
    compute_local[0] = (compute_local[0] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 3)]));
    compute_local[8] = (compute_local[8] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 259)]));
    compute_local[4] = (compute_local[4] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 3)]));
    compute_local[12] = (compute_local[12] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 259)]));
    compute_local[1] = (compute_local[1] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 7)]));
    compute_local[9] = (compute_local[9] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 263)]));
    compute_local[5] = (compute_local[5] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 7)]));
    compute_local[13] = (compute_local[13] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 263)]));
    compute_local[2] = (compute_local[2] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 11)]));
    compute_local[10] = (compute_local[10] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 267)]));
    compute_local[6] = (compute_local[6] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 11)]));
    compute_local[14] = (compute_local[14] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 267)]));
    compute_local[3] = (compute_local[3] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 15)]));
    compute_local[11] = (compute_local[11] + (pad_temp_shared[(((int)get_local_id(0)) + 84)] * kernel_shared[((((int)get_local_id(2)) * 16) + 271)]));
    compute_local[7] = (compute_local[7] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 15)]));
    compute_local[15] = (compute_local[15] + (pad_temp_shared[(((int)get_local_id(0)) + 98)] * kernel_shared[((((int)get_local_id(2)) * 16) + 271)]));
  }
  compute[(((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0)))] = compute_local[0];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 12544)] = compute_local[8];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 14)] = compute_local[4];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 12558)] = compute_local[12];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 196)] = compute_local[1];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 12740)] = compute_local[9];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 210)] = compute_local[5];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 12754)] = compute_local[13];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 392)] = compute_local[2];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 12936)] = compute_local[10];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 406)] = compute_local[6];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 12950)] = compute_local[14];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 588)] = compute_local[3];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 13132)] = compute_local[11];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 602)] = compute_local[7];
  compute[((((((((int)get_group_id(2)) * 896) + ((int)get_group_id(1))) + (((int)get_local_id(2)) * 28)) * 28) + ((int)get_local_id(0))) + 13146)] = compute_local[15];
}
