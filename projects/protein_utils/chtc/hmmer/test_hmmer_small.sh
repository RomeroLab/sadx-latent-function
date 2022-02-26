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


QUERY_BASE=${QUERY_FASTA%%.*}

echo "Running jackhmmer on : ${QUERY_BASE}"

jackhmmer -A "${QUERY_BASE}".sto \
          -o "${QUERY_BASE}".out.txt \
          ${QUERY_FASTA} \
          targetdb.fasta


RESULTS_ARCHIVE="${QUERY_BASE}"_results.tar.gz
echo "Archiving results to : ${RESULTS_ARCHIVE}"

tar cf "${RESULTS_ARCHIVE}" *.sto *.out.txt


## to return results locally we just move this .tar.gz
## to the parent directory and it gets automatically returned
# mv "${RESULTS_ARCHIVE}" ..
## However, it might be too big so we just copy back to staging

# We can only store 1 output file at a time so we
# use this prefix to delete any previous output files
OUTPUT_PREFIX=/staging/dcosta2/hmmer
rm -f "${OUTPUT_PREFIX}"_*
mv ${RESULTS_ARCHIVE} "${OUTPUT_PREFIX}_${RESULTS_ARCHIVE}"

echo "Copied to staging directory"


