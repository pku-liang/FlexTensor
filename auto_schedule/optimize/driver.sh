#!/bin/bash
run() {
    file=${1?"Please Input Filename"}
    from=${2?"Please Input Shape No."}
    target=${3:-"cuda"}
    device=${4:-0}
    timeout=${5:-200}
    parallel=${6:-40}
    to=$((from + 1))

    logfile=${file%.*}_${target}
    python3 ${file} --target ${target} --device ${device} --timeout ${timeout} --parallel ${parallel} -f ${from} -t ${to} --log ${logfile}_schedule.log >& ${logfile}_${from}.log
}

todo_files=(
# optimize_bilinear.py
optimize_block_circulant_matrix.py
# optimize_conv1d.py
# optimize_conv2d_1x1_packed.py
# optimize_conv2d.py
# optimize_conv3d.py
# optimize_conv_transpose1d.py
# optimize_conv_transpose2d.py
# optimize_conv_transpose3d.py
optimize_depthwise_conv2d.py
optimize_dilation_conv2d.py
optimize_gatedPixelCNN.py
# optimize_gemm_conv2d.py
optimize_gemm.py
optimize_gemv.py
# optimize_grouped_conv2d.py
# optimize_mttkrp.py
optimize_pixelCNN.py
# optimize_test_conv.py
optimize_unpooling1d.py
optimize_unpooling2d.py
)


set -x
device=${1:-0}
for file in ${todo_files}; do
    read cnt <<<$(python3 $file --length)
    for ((i = 0; i < cnt; ++i)); do
        run $file $i cuda $device
    done
done

set +x