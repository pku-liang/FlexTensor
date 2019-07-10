#!/bin/bash
run() {
    file=${1?"Please Input Filename"}
    from=${2?"Please Input Shape No."}
    to=$((from + 1))
    for target in llvm cuda; do
        logfile=${file}_${target}
        python3 $file --target ${target} --timeout 120 --parallel 20 -f ${from} -t ${to} --log ${logfile}_schedule.log >&${logfile}_${from}.log
    done
}

# set -x

files=$(echo *.py)

for file in $files; do
    read cnt <<<$(python3 $file --length)
    for ((i = 0; i < cnt; ++i)); do
        run $file $i
    done
done

# set +x
