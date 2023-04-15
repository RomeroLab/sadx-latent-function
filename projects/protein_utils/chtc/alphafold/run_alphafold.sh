#!/bin/bash
#
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'GPU: ' `lspci | grep NVIDIA`
nvidia-smi

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

echo "Creating conda environment"

ENVNAME=alphafold
ENVDIR=$ENVNAME

export PATH
mkdir $ENVDIR

cat alphafold_conda_tar_gz_a? | tar xzf -  -C $ENVDIR
rm alphafold_conda_tar_gz_a?

. $ENVDIR/bin/activate
conda-unpack

echo "patching openmm"
(cd alphafold/lib/python3.9/site-packages; patch -p0 < alphafold/lib/python3.9/site-packages/alphafold/docker/openmm.patch)

echo "Testing out Jax"
python -c "import jax; print(jax.local_devices()[0].platform)"

echo "conda env is setup, now run the script"

QUERY_FASTA=$1
QUERY_BASE=${QUERY_FASTA%%.*}

WORKDIR=work
mkdir -p ${WORKDIR}
cp "${QUERY_FASTA}" "${WORKDIR}"
cp run_alphafold.py "${WORKDIR}"

mkdir alphafold_params

cd alphafold_params
(for letter in {a..d}; do echo $letter; done) | xargs -n1 -P2 bash -c \
        'i=$0; 
url="http://proxy.chtc.wisc.edu/SQUID/dcosta2/alphafold_params/alphafold_params_colab_tar_a${i}"; 
wget ${url}'
cd ..

mkdir -p alphafold/data/params
cat alphafold_params/* | tar xvf - -C alphafold/data/params
rm -r alphafold_params/

cd "${WORKDIR}"
python run_alphafold.py
