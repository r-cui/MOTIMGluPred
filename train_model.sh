#!/bin/bash

GPU=$1
CONFIG=$2  # "ohiot1dm_self", "ohiot1dm_cross"
export CUDA_VISIBLE_DEVICES=${GPU}

python -m experiments.forecast --config_path=experiments/configs/"${CONFIG}".gin build_experiment

i=0
total=($(/bin/ls -d storage/experiments/${CONFIG}/* | wc -l))
for instance in `/bin/ls -d storage/experiments/${CONFIG}/*`; do
    echo "${instance}"
    make run command="${instance}"/command
    i=$((i+1))
    progress=$((i * 100 / total))
    echo "Progress: ${progress}%"
done
