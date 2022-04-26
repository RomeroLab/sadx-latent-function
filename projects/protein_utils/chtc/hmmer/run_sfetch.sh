#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

# setup conda environment
# replace env-name on the right hand side of this line with the name of your conda environment
ENVNAME=hmmer
# if you need the environment directory to be named something other than the environment name, change this line
ENVDIR=$ENVNAME

# these lines handle setting up the environment; you shouldn't have to modify them
export PATH
mkdir $ENVDIR
tar -xzf $ENVNAME.tar.gz -C $ENVDIR
. $ENVDIR/bin/activate

# conda env is setup, now run the script

QUERY_TXT=$1
QUERY_BASE=${QUERY_TXT%%.*}

FULL_UNIREF=false
if [ "$2" = "FULL" ]; then
  FULL_UNIREF=true
fi

STAGING_DIR=/staging/dcosta2
TARGET_DB="${STAGING_DIR}"/uniref100.fasta.gz

WORKDIR=work
mkdir -p ${WORKDIR}
cp "${QUERY_TXT}" "${WORKDIR}"

echo -n "Extracting target database : "
if [ "${FULL_UNIREF}" = true ] ;
then
  echo "using FULL database"
  gunzip -c "${TARGET_DB}" > "${WORKDIR}/targetdb.fasta"
else
  echo "using SMALL database"
  # only get a few fasta sequences
  gunzip -c "${TARGET_DB}" | head -999 > "${WORKDIR}/targetdb.fasta"
fi

cd "${WORKDIR}"
echo "Target database size : " `du -s -h targetdb.fasta`

echo "Creating index file"
esl-sfetch --index targetdb.fasta

RESULTS_FASTA="${QUERY_BASE}".fasta
echo "fetching sequences"
esl-sfetch -o "${RESULTS_FASTA}" -f targetdb.fasta "${QUERY_TXT}"


echo "Current directory    : "
ls -al
echo "Staging directory    : "
ls -al ${STAGING_DIR}

## to return results locally we just move this .tar.gz
## to the parent directory and it gets automatically returned
mv "${RESULTS_FASTA}" ..

cd ..
gzip "${RESULTS_FASTA}"

