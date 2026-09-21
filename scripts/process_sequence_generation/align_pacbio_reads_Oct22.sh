#!/bin/bash

# Run this script in conda environment base. It needs pbmm2 from pacbio
# example run command is below
# (base) ./align_pacbio_reads_Oct22.sh | sh > "../data/scratch/Oct22/pbmm2_output.txt" 2>&1

# location of the parent sequences
fasta_dir="../data" 

# location of the output bam files
output_dir="../data/scratch/Oct22"

# location of the ccs bams and samples
input_dir="../data/scratch/r64120_20221013_213137/3_C01"
samples_file="${input_dir}/00Samples"
demux_dir="${input_dir}/demux"

declare -r -A parent_array=(
  [1]="${fasta_dir}/1-VH.fasta"
  [2]="${fasta_dir}/2-L.fasta"
  [3]="${fasta_dir}/3-VRL.fasta"
)

# samples file is missing a newline at the end so we have to 
# do -n ${barcode} to include the last line. 
# See https://stackoverflow.com/a/12916758/342362
sed '1,2d' "${samples_file}" | while read -r Barcode Parent || [ -n "$Barcode" ];
do 
  echo "Processing : ${Barcode}"
  first_char=${Parent:0:1}
  input_filenames=(${demux_dir}/*${Barcode}--${Barcode}*.bam)
  n_input_filenames=${#input_filenames[@]}
  if [ "${n_input_filenames}" != "1" ]; then
    echo "Found ${n_input_filenames} for ${Barcode}"
    break
  else
    echo pbmm2 align \
            "${parent_array[${first_char}]}" \
            "${input_filenames[0]}" \
            "${output_dir}/${Barcode}.bam" \
            "--preset CCS --log-level INFO --num-threads 8"
  fi
done
