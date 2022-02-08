#!/bin/bash

ROSETTA_SCRATCH_DIR="../../data/scratch/rosetta"
VARIANT_LIST="${ROSETTA_SCRATCH_DIR}/2D_splits/test_variants.txt"
VARIANTS_2D_DIR="${ROSETTA_SCRATCH_DIR}/2D_triple_mutants"
#VARIANTS_2D_TEST_DIR="${ROSETTA_SCRATCH_DIR}/2D_triple_mutants_test_set"

while read variant;
do
  echo ${variant}
  VARIANT_PDB="${variant}.pdb"

  # extract relaxed pdb created by the rosetta simulations
  tar -xzf "${VARIANTS_2D_DIR}"/*"${variant}"*tar.gz \
            ./variant_relaxed.pdb
  mv variant_relaxed.pdb "${VARIANT_PDB}"
done < "${VARIANT_LIST}"

