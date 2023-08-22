#!/bin/bash

# list variants where the output is either missing or less than 1M

# ls -sk is filesize in kilobytes
# We find out which ones are smaller than 1M and make a sorted list

shopt -s extglob

(ls -sk SadA_+([[:alnum:]])_rosetta.tar.gz \
        results/SadA_+([[:alnum:]])_rosetta.tar.gz 2> /dev/null) \
         | awk '{if ($1 < 1000) print $2 }' \
         | cut -d_ -f2 \
         | sort

