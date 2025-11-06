#!/bin/bash

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

ENVNAME=pytorch-gpu
ENVDIR=$ENVNAME

# Setup conda environment by copying bits from squid proxy
export PATH
mkdir $ENVDIR

ENV_TAR_GZ=${ENVNAME}.tar.gz
cat xa*_tar_gz > $ENV_TAR_GZ
rm xa*_tar_gz

tar -xzf $ENV_TAR_GZ -C $ENVDIR
rm $ENV_TAR_GZ
. $ENVDIR/bin/activate

# test python versions and gpu availability
python3 -c "import torch;print('torch version :', torch.__version__)"
if [ -f pytorch_gpu_test_cuda.py ]; then
    python3 pytorch_gpu_test_cuda.py
fi

# set up data
DATA_PKG=EVE.tar.gz
if [ -f ${DATA_PKG} ]; then
    tar zxf ${DATA_PKG}
else
    echo File not found: "${DATA_PKG}" 
    exit 1
fi

cd EVE
mkdir retdir

# 1 is mDHFR
# 0 is BLAT something
PROTEIN_INDEX=1 

# testing sizes
#NUM_SAMPLES_COMPUTE_EVOL_INDICES=100
#MODEL_PARAMS_LOC=./small_train_params.json

# full sizes 
NUM_SAMPLES_COMPUTE_EVOL_INDICES=20000
MODEL_PARAMS_LOC=./EVE/default_model_params.json

python train_VAE.py \
  --MSA_data_folder . \
  --MSA_list msa_list.csv \
  --protein_index ${PROTEIN_INDEX} \
  --MSA_weights_location . \
  --VAE_checkpoint_location . \
  --model_parameters_location ${MODEL_PARAMS_LOC} \
  --training_logs_location .

python compute_evol_indices.py \
  --MSA_data_folder . \
  --MSA_list msa_list.csv \
  --protein_index ${PROTEIN_INDEX} \
  --MSA_weights_location . \
  --VAE_checkpoint_location . \
  --model_parameters_location ${MODEL_PARAMS_LOC} \
  --computation_mode all_singles \
  --all_singles_mutations_folder . \
  --output_evol_indices_location retdir \
  --num_samples_compute_evol_indices ${NUM_SAMPLES_COMPUTE_EVOL_INDICES} \
  --batch_size 2048

mv retdir/*.csv ..
cd ..  
rm -f .bash_history
