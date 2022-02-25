#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=hmmer
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

QUERY_FASTA=$1

TARGET_DB=uniref100_small.fasta.gz
#TARGET_DB=/staging/dcosta2/uniref100.fasta.gz

WORKDIR=work
mkdir -p ${WORKDIR}

cp "${QUERY_FASTA}" "${WORKDIR}"

echo "extracting target database"
gunzip -c "${TARGET_DB}" > ${WORKDIR}/targetdb.fasta

cd "${WORKDIR}"

echo "Target database size: " `du -s -h targetdb.fasta`


QUERY_BASE=`basename ${QUERY_FASTA}`

jackhmmer -A "${QUERY_BASE}".sto \
          -o "${QUERY_BASE}".out.txt \
          ${QUERY_FASTA} \
          targetdb.fasta

tar xcf ../"${QUERY_BASE}"_results.tar.gz *.sto *.out.txt

