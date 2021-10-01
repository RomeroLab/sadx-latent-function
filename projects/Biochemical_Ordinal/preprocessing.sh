#!/bin/bash

for bin_num in `seq 1 8`;
do
    python preprocessing.py \
            -i data/jared_PacBio_data_2/bin${bin_num}.fastq.gz \
            -o data/2D/bin${bin_num}_sequences.txt
done
