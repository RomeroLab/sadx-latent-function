#!/bin/bash


if [[ $# -le 2 ]] ; then
    echo 'Usage: ./$0 PROCESS_NUM OPTIONS_FILE NUM_STRUCTS'
    exit 1
fi

PROCESS_NUM=$1
OPTIONS_FILE=$2
NUM_STRUCTS=$3

ROSETTA_SCRIPTS_EXEC=rosetta_scripts.static.linuxgccrelease
ROSETTA_PATH="."
DATABASE_PATH="database"
GROUP_SERVER_NAME="biocas-00996l.ad.wisc.edu"

#CHTC_CLEANUP=

if [ $(uname -n)  == "${GROUP_SERVER_NAME}" ] ; 
then
  echo "Running on Group Computational Server"
  ROSETTA_PATH="/mnt/scratch/sameer/rosetta/squid"
  DATABASE_PATH="${ROSETTA_PATH}/database"
else
  echo "Running on CHTC"
  # piece together database files and utar
  cat db.tar.bz2.part* | tar -jx  
  
  # clean up database fragments
  rm -f db.tar.bz2.part*
fi

# make output directory for structures
mkdir -p Models

ROSETTA_SCRIPTS_BIN="${ROSETTA_PATH}/${ROSETTA_SCRIPTS_EXEC}"
if [ ! -f "$ROSETTA_SCRIPTS_BIN" ]; then
    echo "Error: Rosetta scripts binary $ROSETTA_SCRIPTS_BIN not found"
    exit 1
fi
chmod +x ${ROSETTA_SCRIPTS_BIN}

OPTIONS_FILE=options.txt
if [ ! -f "$OPTIONS_FILE" ]; then
    echo "Error: Options file $OPTIONS_FILE not found"
    exit 1
fi

if [ -z "$NUM_STRUCTS" ]; then
    echo "Error: Number of structs $NUM_STRUCTS not specified"
    exit 1
fi

ROSETTA3_DB=${DATABASE_PATH} ${ROSETTA_SCRIPTS_BIN} \
    @${OPTIONS_FILE} \
    -nstruct ${NUM_STRUCTS} \

# This file should be returned
# We should name it something appropriate
mv Models/score.sc .

## clean up binary file
#if [ $(uname -n)  != "${GROUP_SERVER_NAME}" ] ; 
#then
#  if [ -z "$CHTC_CLEANUP" ] ;
#  then
#    echo rm -f ${ROSETTA_SCRIPTS_BIN} 
#    echo rm -rf database
#  fi
#fi




