#!/bin/bash
run() {
    file=${1?"Please Input Filename"}
    from=${2?"Please Input Shape No."}
    target=${3?"Please Input Target"}
    to=$((from + 1))

    logfile=${file%.*}_${target}
    python3 ${file} --target ${target} --timeout 120 --parallel 20 -f ${from} -t ${to} --log ${logfile}_schedule.log >&${logfile}_${from}.log
}

set -x

declare -a files
read -a files <<<$(echo *.py)
head=${files[0]}

for file in ${files[@]/$head/}; do
    read cnt <<<$(python3 $file --length)
    for ((i = 0; i < cnt; ++i)); do
        run $file $i cuda
    done
done

for file in ${files}; do
    read cnt <<<$(python3 $file --length)
    for ((i = 0; i < cnt; ++i)); do
        run $file $i llvm
    done
done

set +x
