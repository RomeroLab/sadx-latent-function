#!/bin/bash


if [[ $# -le 2 ]] ; then
    echo 'Usage: ./$0 PROCESS_NUM VARIANT NUM_STRUCTS'
    exit 1
fi

PROCESS_NUM=$1
VARIANT=$2
NUM_STRUCTS=$3

CHAIN="A"

function reportError() {
  if [ $1 -ne 0 ]; then
    echo $2
    exit $1
  fi
}

ROSETTA_SCRIPTS_EXEC=rosetta_scripts.static.linuxgccrelease
ROSETTA_RELAX_EXEC=relax.static.linuxgccrelease
ROSETTA_PATH=`pwd`
DATABASE_PATH="${ROSETTA_PATH}/database"
WORKING_DIR="working"
GROUP_SERVER_NAME="biocas-00996l.ad.wisc.edu"

#CHTC_CLEANUP=

mkdir -p ${WORKING_DIR}

if [ $(uname -n)  == "${GROUP_SERVER_NAME}" ] ; 
then
  echo "Running on Group Computational Server"
  ROSETTA_PATH="/mnt/scratch/sameer/rosetta/squid"
  DATABASE_PATH="${ROSETTA_PATH}/database"

  cp *.params *.pdb *.xml options_*.txt ${WORKING_DIR}

else
  echo "Running on CHTC"
  # piece together database files and untar
  # this goes into a database directory and not working dir
  cat db.tar.bz2.part* | tar -jx 
  
  # clean up database fragments
  rm -f db.tar.bz2.part*

  # copy all input files to working directory
  tar -zxf inputs.tar.gz --directory ${WORKING_DIR}
fi

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
mkdir -p Relax_commandline # for Rosetta output
mkdir -p output # for output returned from chtc

OPTIONS_MUTATE_FILE=options_mutate.txt
if [ ! -f "$OPTIONS_MUTATE_FILE" ]; then
    echo "Error: Options file $OPTIONS_MUTATE_FILE not found"
    exit 1
fi

OPTIONS_DOCK_FILE=options_dock.txt
if [ ! -f "$OPTIONS_DOCK_FILE" ]; then
    echo "Error: Options file $OPTIONS_DOCK_FILE not found"
    exit 1
fi


if [ -z "$NUM_STRUCTS" ]; then
    echo "Error: Number of structs $NUM_STRUCTS not specified"
    exit 1
fi

echo "Making the mutations"
# Make the mutations 
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
	@${OPTIONS_MUTATE_FILE} \
	-nstruct 1

echo "Relaxing the mutation pdb"
# relax the mutated file
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_RELAX_BIN}" \
	-in:file:s Relax_commandline/SadA_NSLeu_Corrected_3701_0002_0001.pdb \
	-in:file:extra_res_fa NEU.params \
	-in:file:extra_res_fa AKG.params  \
	-relax:constrain_relax_to_start_coords \
	-relax:fast \
	-out:path:all Relax_commandline

echo "Copying relaxed structure and scores to output directory"
OUTPUT_YAML=output/info.yaml
echo "starting_struct: " SadA_NSLeu_Corrected_3701_0002  > ${OUTPUT_YAML}
echo "Variant: " "$VARIANT" >> ${OUTPUT_YAML}

cp Relax_commandline/SadA_NSLeu_Corrected_3701_0002_0001_0001.pdb \
	output/variant_relaxed.pdb
mv Relax_commandline/score.sc output/variant_relaxed_score.sc

echo "Docking relaxed structure"
# dock
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    @${OPTIONS_DOCK_FILE} \
    -nstruct ${NUM_STRUCTS} 

echo "copying the best structure and scores to output directory"
# identify and copy the best structure
best_struct=`sort -n -k2 Relax_commandline/score.sc | head -1 | awk  '{print $NF}'`
cp Relax_commandline/"${best_struct}.pdb" output/variant_docked.pdb

# copy the first two lines of the score file
head -2 Relax_commandline/score.sc > output/variant_docked_score.sc
# copy the scores of the best structure
grep "${best_struct}" Relax_commandline/score.sc  >> output/variant_docked_score.sc

echo "tar up output directory"
# tar the output file in the parent directory (above working directory)
# This .tar.gz file will be returned by chtc
tar zcf "../SadA_${VARIANT}_rosetta.tar.gz" -C output .

echo "Done!"



