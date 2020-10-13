#!/bin/env bash
running() {
    beg=$1
    end=$(python3 -c "print($beg + 1)")
    timeout=$(python3 -c "print(($beg + 3) * 2)")
    stdbuf --output=0 --error=0 python3 optimize_gemm.py \
        --target_host "llvm -mtriple=aarch64-linux-android" \
        --host 0.0.0.0 --port 9190 \
        --use_rpc tracker \
        --fcompile ndk \
        --device_key android \
        --target opencl \
        --timeout $timeout \
        --parallel 6 \
        -f $beg -t $end \
        -l fuck.log \
        1>gemm-$beg.log \
        2>gemm-$beg.log
}

start=${1:-0}
stop=${2:-6}

set -x
for ((i = $start; i < $stop; i++)); do
    running $i
done
