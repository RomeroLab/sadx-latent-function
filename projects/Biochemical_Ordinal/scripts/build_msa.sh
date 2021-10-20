#!/bin/bash


fasta_filename="../data/WT/sadA_wt_trimmed.fasta"
stub="sadA"
working_dir="../cache"

sto_filename=${working_dir}/${stub}.sto
out_txt_filename=${working_dir}/${stub}.out.txt
afa_filename=${working_dir}/${stub}.afa
clean_filename=${working_dir}/${stub}_clean.fasta
weights_filename=${working_dir}/${stub}_clean.weights.npy

jackhmmer -A ${sto_filename} \
          -o ${out_txt_filename} \
          ${fasta_filename} \
          /mnt/scratch/databases/uniprot/uniref90.fasta


/home/romeroroot/code/hmmer-3.1b2-linux-intel-x86_64/binaries/esl-reformat \
        -u -o ${afa_filename} afa ${sto_filename}

python3 \
        ~/VAEs/source/make_dataset/jackhmmer_aligned_msa_filter.py \
        -i ${afa_filename} \
        -q ${fasta_filename} \
        -o ${clean_filename}

python3 ~/VAEs/source/reweighting_tools.py \
        -i ${clean_filename} \
        -o ${weights_filename}

rm -f ${sto_filename} ${out_txt_filename} ${afa_filename}

mv ${clean_filename} ${weights_filename} ../data/

