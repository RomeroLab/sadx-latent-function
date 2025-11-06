#!/bin/bash

# Find out where this script is located
# It should be in the bin directory of a protein_utils installation
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

# Find out the python source directory for the corresponding directory
SOURCEDIR=`realpath "$SCRIPTPATH/../.."`

# ADD Source to PYTHONPATH
PYTHONPATH="${SOURCEDIR}:${PYTHONPATH}" python -m protein_utils.external.pdb_to_fasta "$@"
