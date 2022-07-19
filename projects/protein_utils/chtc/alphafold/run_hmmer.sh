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

QUERY_FASTA=$1
QUERY_BASE=${QUERY_FASTA%%.*}

TARGET_DB_FASTA_GZ=$2
TARGET_BASE=${TARGET_DB_FASTA_GZ%.*}

Z_VALUE=$3

WORKDIR=work
mkdir -p ${WORKDIR}
cp "${QUERY_FASTA}" "${WORKDIR}"


gunzip -c "${TARGET_DB_FASTA_GZ}" > "${WORKDIR}/targetdb.fasta"

cd "${WORKDIR}"
echo "Target database size : " `du -s -h targetdb.fasta`

STO_FILENAME="${TARGET_BASE}".sto
TBLOUT_FILENAME="${TARGET_BASE}".tblout.txt 

jackhmmer \
    -o /dev/null \
    -A "${STO_FILENAME}" \
    --F1 0.0005 \
    --F2 0.00005 \
    --F3 0.0000005 \
    --incE 0.0001 \
    -E  0.0001 \
    -N 1 \
    --tblout "${TBLOUT_FILENAME}" \
    -Z "${Z_VALUE}" \
    --noali \
    ${QUERY_FASTA} \
    targetdb.fasta 

mv "${STO_FILENAME}" "${TBLOUT_FILENAME}" ..
