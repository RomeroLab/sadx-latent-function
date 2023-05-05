#!/usr/bin/env bash

# copied from https://github.com/samgelman/RosettaTL/blob/master/htcondor/templates/run.sh 

CODE_FN=code.tar.gz
ENV_FN=rtl.tar.gz
HUB_FN=hub.tar.gz


# create output directory for condor logs early
# not sure exactly when/if this needs to be done
mkdir -p run_output/condor_logs

# echo some HTCondor job information
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "System: $(uname -spo)"
echo "_CONDOR_JOB_IWD: $_CONDOR_JOB_IWD"
echo "Cluster: $CLUSTER"
echo "Process: $PROCESS"
echo "RunningOn: $RUNNINGON"

# this makes it easier to set up the environments, since the PWD we are running in is not $HOME
export HOME=$PWD

# combine all split tar files in the home directory
if [ "$(ls 2>/dev/null -Ubad1 -- *.tar.gz.* | wc -l)" -gt 0 ];
then
  # first get all the unique split tar file prefixes
  declare -A tar_prefixes
  for f in *.tar.gz.*; do
      tar_prefixes[${f%%.*}]=
  done

  # now combine the split tar files for each prefix
  for f in "${!tar_prefixes[@]}"; do
    echo "Combining split files for $f.tar.gz"
    cat "$f".tar.gz.* > "$f".tar.gz
    rm "$f".tar.gz.*
  done
fi

if [ -f "$CODE_FN" ]; then
  echo "Extracting $CODE_FN"
  mkdir code
  tar -zxf $CODE_FN -C code
  rm $CODE_FN
  if [ -f "$HUB_FN" ]; then
    echo "Extracting $HUB_FN"
    tar -zxf "$HUB_FN" -C code
    rm "$HUB_FN"
  fi
fi

# the environment files need to be un-tarred into the "env" directory
# un-tar the environment files
if [ -f "$ENV_FN" ]; then
  echo "Extracting $ENV_FN"
  mkdir env
  tar -zxf $ENV_FN -C env
  rm $ENV_FN
fi


if [ "$(ls 2>/dev/null -Ubad1 -- *.tar.gz | wc -l)" -gt 0 ];
then
  for f in *.tar.gz;
  do
    echo "Extracting $f"
    tar -zxf "$f";
    rm "$f"
  done
fi

echo "Activating Python environment"
export PATH
. env/bin/activate

echo "Conda environment"
ls env/conda-meta/*.json | xargs basename -s .json

echo "Pip environment"
pip list

echo "Launching $RUN_SCRIPT"

python3 -c "import torch; print('torch version : ' + torch.__version__)"
python3 -c "import esm; print('esm version : ' + esm.__version__)"

cd code
python3 esm_model.py --max_epochs 1 --delete_checkpoints

tar zcf ../training_logs.tar.gz output/training_logs

