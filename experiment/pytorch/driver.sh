#!/bin/bash

# set -x

files=$(echo *.py)

for file in $files; do
    python3 $file >& ${file}.log
done

# set +x