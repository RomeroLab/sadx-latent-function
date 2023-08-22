#!/bin/bash

if [[ $# -le 3 ]] ; then
    echo 'Usage: ./$0 PROCESS_NUM START_STRUCT VARIANT NUM_STRUCTS'
    exit 1
fi

PROCESS_NUM=$1
START_STRUCT=$2
VARIANT=$3
NUM_STRUCTS=$4

echo "PROCESS_NUM :" $PROCESS_NUM
echo "START_STRUCT :" $START_STRUCT
echo "VARIANT :" $VARIANT
echo "NUM_STRUCTS :" $NUM_STRUCTS

CHAIN="A"
DEBUG=1


ERROR_SMALL_TAR_GZ=11

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
  cat db.tar.gz.part* | tar -zx 
  
  # clean up database fragments
  rm -f db.tar.gz.part*

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
mkdir -p rmsd # for calculating rmsd 

if [ -z "$NUM_STRUCTS" ]; then
    echo "Error: Number of structs $NUM_STRUCTS not specified"
    exit 1
fi

echo "Copying relaxed structure and scores to output directory"
OUTPUT_YAML=output/info.yaml
echo "starting_struct: " ${START_STRUCT_BASE}  > ${OUTPUT_YAML}
echo "variant: " "$VARIANT" >> ${OUTPUT_YAML}

echo "Making the mutations"
# Make the mutations 
# this also does some sort of relax
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s "${START_STRUCT}" \
    @options_mutate.txt  \
	-nstruct 1 >> rosetta_output.txt

cp mutated_structures/${START_STRUCT_BASE}_0001.pdb \
	output/variant_relaxed.pdb
mv mutated_structures/score.sc output/variant_relaxed_score.sc

##echo "Relaxing the mutation pdb"
## relax the mutated file
#ROSETTA3_DB="${DATABASE_PATH}" \
#	"${ROSETTA_RELAX_BIN}" \
#	-in:file:s mutated_structures/"${START_STRUCT_BASE}"_0001.pdb \
#	-in:file:extra_res_fa NEU.params \
#	-in:file:extra_res_fa AKG.params  \
#	-relax:constrain_relax_to_start_coords \
#	-relax:fast \
#	-out:path:all relaxed_structures >> rosetta_output.txt

#echo "Copying files to output directory" 
#cp relaxed_structures/${START_STRUCT_BASE}_0001_0001.pdb \
#	output/variant_relaxed.pdb
#mv relaxed_structures/score.sc output/variant_relaxed_score.sc


echo "Docking relaxed structure"
# dock
ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    -in:file:s mutated_structures/${START_STRUCT_BASE}_0001.pdb \
    @options_dock.txt \
    -nstruct ${NUM_STRUCTS}  >> rosetta_output.txt

#echo "copying the best structure and scores to output directory"
## identify and copy the best structure
#sort -n -k2 Relax_commandline/score.sc > Relax_commandline/sorted_score.sc
## last field of the first row in the sorted score file
#best_struct=`awk  'NR==1{print $NF}' Relax_commandline/sorted_score.sc`
#cp Relax_commandline/"${best_struct}.pdb" output/variant_docked.pdb
#
## copy the first two lines of the score file
#head -2 Relax_commandline/score.sc > output/variant_docked_score.sc
## copy the scores of the best structure
#head -1 Relax_commandline/sorted_score.sc >> output/variant_docked_score.sc 

# sorting docked structures based on `interface_delta_X`
ls docked_structures/*.pdb > list_of_docked_structures.txt
best_struct_info=`sort -nk 48 docked_structures/score.sc | awk '{print $2, $48, $NF}' | head -1`
IFS=" " read -r best_struct_total_energy \
        best_struct_interface_delta_X \
        best_struct <<< ${best_struct_info}

# copy best structure and all scores to output directory
cp docked_structures/"${best_struct}.pdb" output/
cp docked_structures/score.sc output/docked_score.sc

# calculating rmsd vs energy to best structure
sed "s|NATIVE_COMPARISON_PDB_FILE|./docked_structures/${best_struct}.pdb|g" \
        calculate_rmsd_to_best_model.template.xml > \
        calculate_rmsd_to_best_model.xml

echo "results:" >> ${OUTPUT_YAML}
echo "  best_struct: " ${best_struct} >> ${OUTPUT_YAML}
echo "  best_struct_interface_delta_X: " ${best_struct_interface_delta_X} >> ${OUTPUT_YAML}
echo "  best_struct_total_energy: " ${best_struct_total_energy} >> ${OUTPUT_YAML}

ROSETTA3_DB="${DATABASE_PATH}" \
	"${ROSETTA_SCRIPTS_BIN}" \
    @options_calculate_rmsd_to_best_model.txt >> rosetta_output.txt

cp rmsd/rmsd_to_best_model.sc output/


echo "tar up output directory"
# tar the output file in the parent directory (above working directory)
# This .tar.gz file will be returned by chtc
OUTPUT_TAR_GZ="../SadA_${VARIANT}_rosetta.tar.gz"
OUTPUT_LOGS_TAR_GZ="../SadA_${VARIANT}_logs_rosetta.tar.gz"
tar zcf "${OUTPUT_TAR_GZ}" -C output .

RETVAL=0
if [ -n "$(find "$OUTPUT_TAR_GZ" -prune -size +10000c)" ]; then
    echo "OUTPUT_TAR_GZ File size is larger than 10k"
    echo "Done!"
else
    echo "Sleeping for 10 minutes"
    sleep 600s
    echo "Archiving output and logs for analysis"
    tar zcf "${OUTPUT_LOGS_TAR_GZ}" \
            $(ls rosetta_output.txt ROSETTA_CRASH.log 2> /dev/null)
    RETVAL=${ERROR_SMALL_TAR_GZ}
    reportError $RETVAL "ERROR: OUTPUT_TAR_GZ File size is smaller than 10k"
fi

exit ${RETVAL}






