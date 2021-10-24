#!/bin/bash

## Copy latest parameters from results directory to current directory

set -x

latest_file=`ls results/parameters_J_*.bin | tail -1`
latest_iter_num=`basename -s .bin "$latest_file" | awk -F_ '{print $3}'`

arma2ascii -P results/parameters_J_${latest_iter_num}.bin \
           -p results/parameters_h_${latest_iter_num}.bin

mv results/parameters_${latest_iter_num}.txt .
    


