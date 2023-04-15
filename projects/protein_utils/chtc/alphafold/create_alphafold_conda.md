
### Creating the conda environment for alphafold to run in

The challenge here was to install everything in a way that `conda pack` did not complain.

```shell
## Remove old environment if we are not starting clean
#conda env remove -n alphafold

# we have to fix python=3.9. 
# openmm=7.5.1 because alphafold wants to patch it
# pdbfixer can be anything
# openmm will want to download cudatoolkit and numpy and a bunch of stuff
# nump=1.21.6 is fixed so that pip doesn't override it later. If pip 
#    overrides it later then conda pack won't work
conda create -qy -n alphafold \
        --channel conda-forge \
        python=3.9 \
        openmm=7.5.1 \
        pdbfixer \
        numpy=1.21.6

# activate the environment
conda activate alphafold


# download the alphafold repo in the home directory
cd ~
git clone --branch main https://github.com/deepmind/alphafold alphafold

# make sure that pip doesn't uninstall anything that conda installed
pip3 install -r ~/alphafold/requirements.txt
# we need to download a cuda version of jax. somehow pip installed the 
# wrong version of jaxlib above (not compatible with jax)
# so we reinstall the correct version below
pip3 install --upgrade jaxlib==0.3.25+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# now install the actual alphafold package
pip3 install --no-dependencies ./alphafold
# since we aren't making ssl connections we could probably skip this?
#pip3 install --upgrade pyopenssl

# patch openmm
# patching openmm doesn't work with conda pack so we will do this  
# in the run script after we unpack
#(cd ~/miniconda3/envs/alphafold/lib/python3.9/site-packages; patch -p0 < ~/alphafold/docker/openmm.patch)

# install stereo_chemical properties file
# this doesn't have a problem with conda pack
(cd ~/miniconda3/envs/alphafold/lib/python3.9/site-packages/alphafold/common; wget https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt)

pip3 install ipython matplotlib py3dmol

# check that conda environment works
conda list
# jax should load and default to CPU after complaining about gpu not found
# however on a machine with a GPU it should find the GPU
python -c "import jax; print(jax.local_devices()[0].platform)"

conda deactivate

# install conda-pack if we dont already have it
# conda install -c conda-forge conda-pack

# delete old archive if it exists
rm -f alphafold.tar.gz

conda pack -n alphafold
# check that the resulting tarball is created properly
tar tvf alphafold.tar.gz
du -sh alphafold.tar.gz
split -b 950M alphafold.tar.gz alphafold_conda_tar_gz_
rm -f /squid/dcosta2/py39/alphafold_conda_tar_gz_a*
mv alphafold_conda_tar_gz_a* /squid/dcosta2/py39

# Downloading the Alphafold parameters
# parameters are around 3.8GB
# split and save to squid
wget https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-12-06.tar
split -b 950M alphafold_params_colab_2022-12-06.tar  /squid/dcosta2/alphafold_params/alphafold_params_colab_tar_
mv alphafold_params_colab_tar_a* /squid/dcosta2/alphafold_params


# delete files that were split and moved to squid
rm alphafold.tar.gz alphafold_params_colab_2022-12-06.tar

```
