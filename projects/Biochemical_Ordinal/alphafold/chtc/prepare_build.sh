wget https://raw.githubusercontent.com/deepmind/alphafold/main/requirements.txt

condor_submit -i build.sub

# now run build_script.sh

mkdir -p /squid/dcosta2/py39
mv alphafold_packages.tar.gz /squid/dcosta2/py39
