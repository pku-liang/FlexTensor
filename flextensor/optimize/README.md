# Testing Operators for Mali GPU Optimized by FlexTensor 

[TOC]

## Preparation

### [Installation of TVM](https://tvm.apache.org/docs/install/)

Note: Only need to turn on `USE_LLVM` in `config.cmake`.

### [Installation of TVM Java Frontend](https://github.com/apache/incubator-tvm/blob/main/jvm/README.md)

### [Installation&Setup of TVM-RPC App](https://github.com/apache/incubator-tvm/blob/main/apps/android_rpc/README.md)



## Test

1. Setup environment:

   1. Startup TVM RPC tracker and TVM RPC app.
   2. Export `/path/to/FlexTensor` to `PYTHONPATH`.

2. Open `FlexTensor/flextensor/optimize`.

3. Run the following commands to test optimized operators: gemm, conv1d, conv2d (suppose tracker's ip:port is 0.0.0.0:9190):

   ```shell
   python3 optimize_gemm.py \
       --target_host "llvm -mtriple=aarch64-linux-android" \
       --host 0.0.0.0 --port 9190 \
       --use_rpc tracker \
       --fcompile ndk \
       --device_key android \
       --target opencl \
       --test gemm-config.log
   
   python3 optimize_conv1d.py \
       --target_host "llvm -mtriple=aarch64-linux-android" \
       --host 0.0.0.0 --port 9190 \
       --use_rpc tracker \
       --fcompile ndk \
       --device_key android \
       --target opencl \
       --test conv1d-config.log
   
   python3 optimize_conv2d.py \
       --target_host "llvm -mtriple=aarch64-linux-android" \
       --host 0.0.0.0 --port 9190 \
       --use_rpc tracker \
       --fcompile ndk \
       --device_key android \
       --target opencl \
       --test conv2d-config.log
   ```

   Then you will see the generated kernels and evaluation results.

