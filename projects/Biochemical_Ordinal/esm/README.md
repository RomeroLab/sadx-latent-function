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

```
# installing rtl2
mamba env create -f env_2.yml # conda is too slow
conda activate rtl2

# something missing from env_2.yml that is needed in one of the imports
pip install finetuning-scheduler
```

## Notes
