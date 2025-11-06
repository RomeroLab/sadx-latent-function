#!/bin/bash

set -x

output_dir="outputs"

mkdir -p $output_dir

rm -f $output_dir/score_only/*.npz


mpnn_path="./ProteinMPNN"


path_to_PDB="./inputs/pdbs/SadA_NSLeu_Corrected_3701_best_structure_0044.pdb"

path_to_fasta="./inputs/seqs/SadA.fa"

chains_to_design="A"


python ${mpnn_path}/protein_mpnn_run.py \
	--path_to_fasta $path_to_fasta \
	--pdb_path $path_to_PDB \
	--pdb_path_chains "$chains_to_design" \
        --out_folder $output_dir \
	--num_seq_per_target 10 \
        --sampling_temp "0.1" \
	--score_only 1 \
        --seed 13 \
        --batch_size 1 | tee $output_dir/mpnn_rosetta_struct_output.txt

grep "Score for" $output_dir/mpnn_rosetta_struct_output.txt \
	| awk '{print $3 ",", $7, $9}' | sed 's/,$//' \
	> $output_dir/mpnn_rosetta_struct_output_parsed.txt

