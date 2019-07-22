#!/bin/bash

set -x

files=*.py

for file in $files; do
    logfile=${file%.*}
    python3 $file >& ${logfile}.log
done

set +x