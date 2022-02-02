# in interative session
tar -xzf python39.tar.gz
export PATH=$PWD/python/bin:$PATH
python3 --version
which python3
mkdir packages
python3 -m pip install --target=$PWD/packages -r requirements.txt
tar -czf alphafold_packages.tar.gz packages/

