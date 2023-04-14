#!/bin/bash
#
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'GPU: ' `lspci | grep NVIDIA`
nvidia-smi

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

ENVNAME=alphafold
ENVDIR=$ENVNAME

export PATH
mkdir $ENVDIR

cat alphafold_conda_tar_gz_a? | tar xzf -  -C $ENVDIR
rm alphafold_conda_tar_gz_a?

. $ENVDIR/bin/activate

python -c "import jax; print(jax.local_devices()[0].platform)"

# conda env is setup, now run the script

QUERY_FASTA=$1
QUERY_BASE=${QUERY_FASTA%%.*}

WORKDIR=work
mkdir -p ${WORKDIR}
cp "${QUERY_FASTA}" "${WORKDIR}"
cp run_alphafold.py "${WORKDIR}"

mkdir alphafold_params

cd alphafold_params
wget http://proxy.chtc.wisc.edu/SQUID/dcosta2/alphafold_params/alphafold_params_colab_tar_aa
wget http://proxy.chtc.wisc.edu/SQUID/dcosta2/alphafold_params/alphafold_params_colab_tar_ab
wget http://proxy.chtc.wisc.edu/SQUID/dcosta2/alphafold_params/alphafold_params_colab_tar_ac
wget http://proxy.chtc.wisc.edu/SQUID/dcosta2/alphafold_params/alphafold_params_colab_tar_ad
cd ..

mkdir -p alphafold/data/params
cat alphafold_params/* | tar xvf - -C alphafold/data/params
rm -r alphafold_params/

cd "${WORKDIR}"
python alphafold.py
