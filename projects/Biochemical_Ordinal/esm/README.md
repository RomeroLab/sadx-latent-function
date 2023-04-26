Running esm fine tunning by copying code from [Sam Gelman's RosettaTL repository](https://github.com/samgelman/RosettaTL) and then modifying

## Rosetta transfer learning environment on group server

```shell
cd ~/software
git clone git@github.com:samgelman/RosettaTL.git

cd RosettaTL
mamba env create -f env_2.yml # conda is too close
conda activate rtl2

# something missing from env_2.yml that is needed in one of the imports
pip install finetuning-scheduler
```

## Notes
