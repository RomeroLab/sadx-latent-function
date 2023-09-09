#!/bin/bash

set -x 
../create_variant_xml.sh I71V.D157G.R172H A > wt_to_parent1.xml

ROSETTA_PATH="/mnt/scratch/sameer/rosetta/squid"
DATABASE_PATH="${ROSETTA_PATH}/database"

ROSETTA_SCRIPTS_EXEC=rosetta_scripts.static.linuxgccrelease
ROSETTA_SCRIPTS_BIN="${ROSETTA_PATH}/${ROSETTA_SCRIPTS_EXEC}"

START_STRUCT="SadA_NSLeu_Corrected_3701_0002.pdb"
START_STRUCT_NO_PATH="${START_STRUCT##*/}"
START_STRUCT_BASE="${START_STRUCT_NO_PATH%.pdb}"

ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s "${START_STRUCT}" \
    -parser:protocol wt_to_parent1.xml \
    @options_mutate.txt  \
	&> rosetta_output.txt


cp mutated_structures/${START_STRUCT_BASE}_0001.pdb 1VH.pdb
../create_variant_xml.sh F152L A > parent1_to_parent2.xml
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s 1VH.pdb \
    -parser:protocol parent1_to_parent2.xml \
    @options_mutate.txt  \
	&> rosetta_output.txt


cp mutated_structures/1VH_0001.pdb 2L.pdb
../create_variant_xml.sh I38V.Q233R.F261L A > parent2_to_parent3.xml
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s 2L.pdb \
    -parser:protocol parent2_to_parent3.xml \
    @options_mutate.txt  \
	&> rosetta_output.txt

cp mutated_structures/2L_0001.pdb 3VRL.pdb

