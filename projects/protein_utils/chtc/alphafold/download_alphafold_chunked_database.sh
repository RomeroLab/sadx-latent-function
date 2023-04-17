#!/bin/bash


DB_ROOT_PATH=https://storage.googleapis.com/alphafold-colab/latest

for i in {1..62}
do
  wget ${DB_ROOT_PATH}/uniref90_2022_01.fasta.${i}
  gzip uniref90_2022_01.fasta.${i} &
done
for i in {1..17}
do
  wget ${DB_ROOT_PATH}/bfd-first_non_consensus_sequences.fasta.${i}
  gzip bfd-first_non_consensus_sequences.fasta.${i} &
done
for i in {1..120}
do
  wget ${DB_ROOT_PATH}/mgy_clusters_2022_05.fasta.${i}
  gzip mgy_clusters_2022_05.fasta.${i} &
done
for i in {1..101}
do
  wget ${DB_ROOT_PATH}/uniref90_2022_01.fasta.${i}
  gzip uniref90_2022_01.fasta.${i} &
done



