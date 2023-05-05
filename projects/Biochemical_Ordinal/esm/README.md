Running esm fine tunning by copying code from [Sam Gelman's RosettaTL repository](https://github.com/samgelman/RosettaTL) and then modifying

## Rosetta transfer learning environment on group server

```shell
cd ~/software
git clone git@github.com:samgelman/RosettaTL.git

cd RosettaTL

conda activate base

mamba env create -f env.yml

conda pack -n rtl

du -sh rtl.tar.gz

split -b 950M rtl.tar.gz /squid/dcosta2/py39/rtl.tar.gz.

```

```shell
# install torch.hub esm cache

mkdir -p output/esm_pretrained_models/checkpoints

pushd output/esm_pretrained_models
wget "https://github.com/facebookresearch/esm/archive/main.zip"
unzip main.zip
popd

pushd output/esm_pretrained_models/checkpoints 
wget "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt"
wget "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t6_8M_UR50D-contact-regression.pt"
wget "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt"
wget "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t12_35M_UR50D-contact-regression.pt"
popd

tar zcf /squid/dcosta2/esm/hub.tar.gz output
rm -rf output

```
# installing rtl2
mamba env create -f env_2.yml # conda is too slow
conda activate rtl2

# something missing from env_2.yml that is needed in one of the imports
pip install finetuning-scheduler
```

## Notes
