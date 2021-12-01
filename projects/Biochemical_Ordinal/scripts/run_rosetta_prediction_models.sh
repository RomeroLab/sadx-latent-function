#!/bin/bash

model_stats_dir=../data/scratch/model_stats
model_name_arr=(LinearRosettaEnergyPrediction)
num_variants_arr=(1000 10000 100000)
embed_ncomp_arr=(0 8)
epochs=50

for model_name in ${model_name_arr[@]}; 
do
  for num_variants in ${num_variants_arr[@]};
  do
    for embed_ncomp in ${embed_ncomp_arr[@]};
    do
      python3 trainer_module.py \
        -m ${model_name} \
        -n ${num_variants} \
        -e ${epochs} \
        -c ${embed_ncomp} \
        -o "${model_stats_dir}/${model_name}_${num_variants}_${embed_ncomp}.pkl"
      if (( embed_ncomp > 0 )) ; then
        python3 trainer_module.py \
          -m ${model_name} \
          -n ${num_variants} \
          -e ${epochs} \
          -c ${embed_ncomp} \
          -u  \
          -o "${model_stats_dir}/${model_name}_${num_variants}_${embed_ncomp}_trained.pkl"
      fi
    done
  done
done
