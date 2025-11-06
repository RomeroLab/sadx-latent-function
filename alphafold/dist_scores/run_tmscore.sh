#!/bin/bash

# Results for comparison to starting structure
COMPARISON_TO_REF_RESULTS="results/ref_comparison.txt" 
# Results for comparison to repeated rosetta simulation (2nd try on test set)
COMPARISON_TO_REPEAT_RESULTS="results/repeat_comparison.txt"

# This is the modified TMscore binary that has outfmt==3. 
# This output format includes the GDT scores as well as TMscores
# in a compact format
TMSCORE_BIN="../../bin/TMscore"

ROSETTA_SCRATCH_DIR="../../data/scratch/rosetta"
VARIANT_LIST="${ROSETTA_SCRATCH_DIR}/2D_splits/test_variants.txt"
VARIANTS_2D_DIR="${ROSETTA_SCRATCH_DIR}/2D_triple_mutants"
VARIANTS_2D_TEST_DIR="${ROSETTA_SCRATCH_DIR}/2D_triple_mutants_test_set"

GLOBAL_COMPARISON_PDB="SadA_NSLeu_Corrected_3701_0002.pdb"

TMP_DIR=tmp
mkdir -p "${TMP_DIR}"
GLOBAL_COMPARISON_TMP_PDB="${TMP_DIR}/${GLOBAL_COMPARISON_PDB}"
cp "${GLOBAL_COMPARISON_PDB}" "${GLOBAL_COMPARISON_TMP_PDB}"

rm -f "${COMPARISON_TO_REF_RESULTS}" "${COMPARISON_TO_REPEAT_RESULTS}"

while read variant;
do
  echo ${variant}
  VARIANT_PDB="${TMP_DIR}/${variant}_1st.pdb"
  VARIANT_PDB2="${TMP_DIR}/${variant}_2nd.pdb"

  # extract relaxed pdb created by the rosetta simulations
  tar -xzf "${VARIANTS_2D_DIR}"/*"${variant}"*tar.gz \
            -C "${TMP_DIR}" ./variant_relaxed.pdb
  mv "${TMP_DIR}/variant_relaxed.pdb" "${VARIANT_PDB}"
  ${TMSCORE_BIN} "${GLOBAL_COMPARISON_TMP_PDB}" "${VARIANT_PDB}" \
          -outfmt 3  | sed '/^#/d' >> "${COMPARISON_TO_REF_RESULTS}"

  # extract the second rosetta simulation of the same variant
  tar -xzf "${VARIANTS_2D_TEST_DIR}"/*"${variant}"*tar.gz \
          -C "${TMP_DIR}" ./variant_relaxed.pdb
  mv "${TMP_DIR}/variant_relaxed.pdb" "${VARIANT_PDB2}"
  ${TMSCORE_BIN} "${VARIANT_PDB2}" "${VARIANT_PDB}" \
          -outfmt 3  | sed '/^#/d' >> "${COMPARISON_TO_REPEAT_RESULTS}"

  # clean up two variant files
  rm -f ${VARIANT_PDB} "${VARIANT_PDB2}"

done < "${VARIANT_LIST}"

# end clean up
rm -rf "${TMP_DIR}"
