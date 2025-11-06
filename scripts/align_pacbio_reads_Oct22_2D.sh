#!/bin/bash

# Run this script in conda environment base. It needs pbmm2 from pacbio
# example run command is below
# (base) ./align_pacbio_reads_Oct22.sh | sh > "../data/scratch/Oct22/pbmm2_output.txt" 2>&1

# location of the parent sequences
fasta_dir="../data/2D" 

# location of the output bam files
output_dir="../data/scratch/Oct22_2D"

# location of the ccs bams and samples
input_dir="../data/scratch/r64247_20210830_223032/2_B01"
samples_file="${input_dir}/00Samples"
demux_dir="${input_dir}/demux"

parent_fasta=${fasta_dir}/2D.fasta

# use [ ! -z "$Barcode" ] to skip the last empty line
# sed 1d deletes the header
sed '1d' "${samples_file}" | while read -r Sample Barcode && [ !  -z "$Barcode" ];
do 
  echo "Processing : ${Barcode} , Sample : ${Sample}"
  input_filenames=(${demux_dir}/*${Barcode}.bam)
  n_input_filenames=${#input_filenames[@]}
  if [ "${n_input_filenames}" != "1" ]; then
    echo "Found ${n_input_filenames} for ${Barcode}"
    break
  else
    echo pbmm2 align \
            "${parent_fasta}" \
            "${input_filenames[0]}" \
            "${output_dir}/${Sample}.bam" \
            "--preset CCS --log-level INFO --num-threads 8"
  fi
done
