
### create hmmer package conda environment 

First activate `conda` in the `base` environment and then run the following

```shell

# create new environment and install packages
conda create -n hmmer
conda activate hmmer
conda install -c bioconda hmmer

conda deactivate # back to base environment

# next command is only if conda-pack isn't installed in base
conda install -c conda-forge conda-pack 

# bundle up new conda environment that was created
conda pack -n hmmer
chmod 644 hmmer.tar.gz 
ls -sh hmmer.tar.gz 

```
