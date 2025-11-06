## Conda environment on CHTC

We follow the conda-pack method in the [CHTC how-to](https://chtc.cs.wisc.edu/uw-research-computing/conda-installation)



## Activate conda environment

```shell
eval "$(/home/dcosta2/miniconda3/bin/conda shell.bash hook)" 
conda activate base
```

## Installation on submit server

```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
# type no when it asks to save init commands to .bashrc

# activate conda base manully with eval and activate 
# then install conda-pack in the base environment
conda install -c conda-forge conda-pack # if not installed

```
