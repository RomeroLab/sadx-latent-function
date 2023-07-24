#!/bin/bash

if [[ $# -le 3 ]] ; then
    echo 'Usage: ./$0 PROCESS_NUM START_STRUCT VARIANT NUM_STRUCTS'
    exit 1
fi

PROCESS_NUM=$1
START_STRUCT=$2
VARIANT=$3
NUM_STRUCTS=$4

CHAIN="A"

function reportError() {
  if [ $1 -ne 0 ]; then
    echo $2
    exit $1
  fi
}


# https://stackoverflow.com/questions/2664740/extract-file-basename-without-path-and-extension-in-bash/
START_STRUCT_NO_PATH="${START_STRUCT##*/}"
START_STRUCT_BASE="${START_STRUCT_NO_PATH%.pdb}"

ROSETTA_SCRIPTS_EXEC=rosetta_scripts.static.linuxgccrelease
ROSETTA_RELAX_EXEC=relax.static.linuxgccrelease
ROSETTA_PATH=`pwd`
DATABASE_PATH="${ROSETTA_PATH}/database"
WORKING_DIR="working"
GROUP_SERVER_NAME="biocas"

#CHTC_CLEANUP=

mkdir -p ${WORKING_DIR}

if [[ $(uname -n)  == ${GROUP_SERVER_NAME}* ]] ; 
then
  echo "Running on Group Computational Server"
  ROSETTA_PATH="/mnt/scratch/sameer/rosetta/squid"
  DATABASE_PATH="${ROSETTA_PATH}/database"

  #cp *.params *.pdb *.xml options_*.txt ${WORKING_DIR}

else
  echo "Running on CHTC"
  # piece together database files and untar
  # this goes into a database directory and not working dir
  cat db.tar.bz2.part* | tar -jx 
  
  # clean up database fragments
  rm -f db.tar.bz2.part*

fi

#copy all input files to working directory
tar -zxf inputs.tar.gz --directory ${WORKING_DIR}

ROSETTA_SCRIPTS_BIN="${ROSETTA_PATH}/${ROSETTA_SCRIPTS_EXEC}"
ROSETTA_RELAX_BIN="${ROSETTA_PATH}/${ROSETTA_RELAX_EXEC}"
if [ ! -f "$ROSETTA_SCRIPTS_BIN" ]; then
    echo "Error: Rosetta scripts binary $ROSETTA_SCRIPTS_BIN not found"
    exit 1
fi
chmod +x ${ROSETTA_SCRIPTS_BIN}

if [ ! -f "$ROSETTA_RELAX_BIN" ]; then
    echo "Error: Rosetta relax binary $ROSETTA_RELAX_BIN not found"
    exit 1
fi
chmod +x ${ROSETTA_RELAX_BIN}

echo "Creating Variant XML file"
# create the variant file to mutate
chmod +x create_variant_xml.sh
./create_variant_xml.sh "$VARIANT" "$CHAIN" > "${WORKING_DIR}"/SadA_mutate.xml
reportError $? "Create Variant XML script failed"

echo "Making working directories"
# Now let's move to the working directory
cd working
# make output directory for structures
mkdir -p mutated_structures # for output of rosetta scripts
mkdir -p relaxed_structures # for Rosetta relax output
mkdir -p docked_structures # for Rosetta relax output
mkdir -p output # for output returned from chtc
mkdir -p grid_cache_dir # for storing grid scores

if [ -z "$NUM_STRUCTS" ]; then
    echo "Error: Number of structs $NUM_STRUCTS not specified"
    exit 1
fi

echo "Copying relaxed structure and scores to output directory"
OUTPUT_YAML=output/info.yaml
echo "starting_struct: " ${START_STRUCT_BASE}  > ${OUTPUT_YAML}
echo "Variant: " "$VARIANT" >> ${OUTPUT_YAML}

echo "Making the mutations"
# Make the mutations 
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s "${START_STRUCT}" \
    @options_mutate.txt  \
	-nstruct 1


# we are not relaxing the structure for now
#echo "Relaxing the mutation pdb"
## relax the mutated file
#ROSETTA3_DB="${DATABASE_PATH}" \
#	"${ROSETTA_RELAX_BIN}" \
#	-in:file:s Models/"${START_STRUCT_BASE}"_0001.pdb \
#	-in:file:extra_res_fa NEU.params \
#	-in:file:extra_res_fa AKG.params  \
#	-relax:constrain_relax_to_start_coords \
#	-relax:fast \
#	-out:path:all Relax_commandline
#
#
#echo "Copying files to output directory" 
#cp Relax_commandline/${START_STRUCT_BASE}_0001_0001.pdb \
#	output/variant_relaxed.pdb
#mv Relax_commandline/score.sc output/variant_relaxed_score.sc


echo "Docking relaxed structure"
# dock
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s mutated_structures/${START_STRUCT_BASE}_0001.pdb \
    @options_dock.txt \
    -nstruct ${NUM_STRUCTS} 

echo "Done for now"
exit

echo "copying the best structure and scores to output directory"
# identify and copy the best structure
sort -n -k2 Relax_commandline/score.sc > Relax_commandline/sorted_score.sc
# last field of the first row in the sorted score file
best_struct=`awk  'NR==1{print $NF}' Relax_commandline/sorted_score.sc`
cp Relax_commandline/"${best_struct}.pdb" output/variant_docked.pdb

# copy the first two lines of the score file
head -2 Relax_commandline/score.sc > output/variant_docked_score.sc
# copy the scores of the best structure
head -1 Relax_commandline/sorted_score.sc >> output/variant_docked_score.sc 

echo "tar up output directory"
# tar the output file in the parent directory (above working directory)
# This .tar.gz file will be returned by chtc
tar zcf "../SadA_${VARIANT}_rosetta.tar.gz" -C output .

echo "Done!"



