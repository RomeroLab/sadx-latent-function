#!/bin/bash


DB_ROOT_PATH=https://storage.googleapis.com/alphafold-colab/latest

for i in {1..62}
do
  echo wget ${DB_ROOT_PATH}/uniref90_2022_01.fasta.${i}
done
for i in {1..17}
do
  echo wget ${DB_ROOT_PATH}/bfd-first_non_consensus_sequences.fasta.${i}
done
for i in {1..120}
do
  echo wget ${DB_ROOT_PATH}/mgy_clusters_2022_05.fasta.${i}
done


