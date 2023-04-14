
```shell
#conda env remove -n alphafold
conda create -n alphafold python=3.9
conda activate alphafold

#pip3 install py3dmol
conda install -c nvidia cudatoolkit cudnn


#conda install -c conda-forge pdbfixer openmm=7.5.1 ipython 

#conda install -qy conda==4.13.0
# extra stuff for interactively using alphafold
#conda install -qy -c conda-forge python=3.9 openmm=7.5.1 pdbfixer notebook ipython
#ipython kernel install --name "alphafold-conda" --user
#conda install -qy -c conda-forge python=3.9 openmm=7.5.1 pdbfixer 
#

cd ~
git clone --branch main https://github.com/deepmind/alphafold alphafold
pip3 install -r ~/alphafold/requirements.txt
pip3 install jaxlib==0.3.25
#pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


pip3 install --no-dependencies ./alphafold
#pip3 install --upgrade pyopenssl

# patch openmm
#(cd ~/miniconda3/envs/alphafold/lib/python3.9/site-packages; patch -p0 < ~/alphafold/docker/openmm.patch)
# stereo_chemical properties
(cd ~/miniconda3/envs/alphafold/lib/python3.9/site-packages/alphafold/common; wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt)

conda list

conda deactivate

# install conda-pack if we dont already have it
# conda install -c conda-forge conda-pack

# delete old archive if it exists
rm -f alphafold.tar.gz

conda pack -n alphafold
# check that the resulting tarball is created properly
tar tvf alphafold.tar.gz
split -b 950M alphafold.tar.gz alphafold_conda_tar_gz_
rm -f /squid/dcosta2/py39/alphafold_conda_tar_gz_a*
mv alphafold_conda_tar_gz_a* /squid/dcosta2/py39

# parameter are around 3.8GB
wget https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar
split -b 950M alphafold_params_colab_2022-12-06.tar  /squid/dcosta2/alphafold_params/alphafold_params_colab_tar_
mv alphafold_params_colab_tar_a* /squid/dcosta2/alphafold_params

rm alphafold.tar.gz alphafold_params_colab_2022-12-06.tar

```
