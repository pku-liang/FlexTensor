#define USER_DEFINED_COMPUTE_SIZE_COMPUTE  xxx
#define USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER  xxx
#define USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER1  xxx

__kernel void default_function_kernel0(__global float* restrict compute, __global float* restrict placeholder, __global float* restrict placeholder1) {
  // declare use local memory
  local float local_compute[USER_DEFINED_COMPUTE_SIZE_COMPUTE];
  local float local_placeholder[USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER];
  local float local_placeholder1[USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER1];

  // read data to local memory
  event_t evt[3];
  evt[0] = async_work_group_copy(local_compute, compute, USER_DEFINED_COMPUTE_SIZE_COMPUTE, 0);
  evt[1] = async_work_group_copy(local_placeholder, placeholder, USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER, 0);
  evt[2] = async_work_group_copy(local_placeholder1, placeholder1, USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER1, 0);
  // barrier
  wait_group_events(2, evt);

  // compute pipeline
  for (int b_inner = 0; b_inner < 8; ++b_inner) {
    for (int c = 0; c < 1024; ++c) {
      for (int h = 0; h < 7; ++h) {
        for (int w = 0; w < 7; ++w) {
          local_compute[((((b_inner * 50176) + (c * 49)) + (h * 7)) + w)] = 0.000000e+00f;
          for (int rc = 0; rc < 1024; ++rc) {
            for (int rw = 0; rw < 3; ++rw) {
              // FPGA pipeline
              __attribute_((xcl_pipeline_loop)){
              for (int rh = 0; rh < 3; ++rh) {
                local_compute[((((b_inner * 50176) + (c * 49)) + (h * 7)) + w)] = (local_compute[((((b_inner * 50176) + (c * 49)) + (h * 7)) + w)] + ((float)(((1 <= ((h * 2) + rh)) && (1 <= ((w * 2) + rw))) ? local_placeholder[(((((((b_inner * 200704) + (rc * 196)) + (h * 28)) + (rh * 14)) + (w * 2)) + rw) - 15)] : 0.000000e+00f) * local_placeholder1[((((c * 9216) + (rc * 9)) + (rh * 3)) + rw)]));
              }
              }
            }
          }
        }
      }
    }
  }


  // write data back to global memory
  evt[0] = async_work_group_copy(compute, local_compute, USER_DEFINED_COMPUTE_SIZE_COMPUTE, 0);
  evt[1] = async_work_group_copy(placeholder, local_placeholder, USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER, 0);
  evt[2] = async_work_group_copy(placeholder1, local_placeholder1, USER_DEFINED_COMPUTE_SIZE_PLACEHOLDER1, 0);
  // barrier
  wait_group_events(2, evt);
}