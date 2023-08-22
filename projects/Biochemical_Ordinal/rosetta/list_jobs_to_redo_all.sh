#!/bin/bash

# list variants where the output is either missing or less than 1M

# ls -sk is filesize in kilobytes
# We find out which ones are greater than 1M and make a sorted list
# Then we print out which files are unique to the original variant list

shopt -s extglob

comm <(sort SadA_NSLeu_Corrected_3701_best_structure_0044.single_mutants.txt) \
     <(ls -sk SadA_+([[:alnum:]])_rosetta.tar.gz \
            results/SadA_+([[:alnum:]])_rosetta.tar.gz \
        | awk '{if ($1 > 1000) print $2 }' \
        | cut -d_ -f2 | sort) \
     -3  
