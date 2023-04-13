
```shell
conda create -n alphafold python=3.8
conda activate alphafold

wget https://raw.githubusercontent.com/deepmind/alphafold/main/requirements.txt
pip3 install -r requirements.txt
rm requirements.txt

pip3 install notebook ipython seaborn
ipython kernel install --name "alphafold-conda" --user

conda deactivate

# install conda-pack if we dont already have it
# conda install -c conda-forge conda-pack

conda pack -n alphafold
mv alphafold.tar.gz 
mv alphafold.tar.gz /squid/dcosta2/conda_alphafold_py38.tar.gz

```
