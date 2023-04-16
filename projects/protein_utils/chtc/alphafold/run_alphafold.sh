#!/bin/bash
#
echo 'Date: ' `date`
echo 'Host: ' `hostname`
echo 'System: ' `uname -spo`
echo 'GPU: ' `lspci | grep NVIDIA`
nvidia-smi

# have job exit if any command returns with non-zero exit status (aka failure)
set -e

echo "Creating conda environment"

ENVNAME=alphafold
ENVDIR=$ENVNAME

export PATH
mkdir $ENVDIR

cat alphafold_conda_tar_gz_a? | tar xzf -  -C $ENVDIR
rm alphafold_conda_tar_gz_a?

. $ENVDIR/bin/activate
conda-unpack

echo "patching openmm"
(cd alphafold/lib/python3.9/site-packages ; patch -p0 <<EOF
Index: simtk/openmm/app/topology.py
===================================================================
--- simtk.orig/openmm/app/topology.py
+++ simtk/openmm/app/topology.py
@@ -356,19 +356,35 @@
         def isCyx(res):
             names = [atom.name for atom in res._atoms]
             return 'SG' in names and 'HG' not in names
+        # This function is used to prevent multiple di-sulfide bonds from being
+        # assigned to a given atom. This is a DeepMind modification.
+        def isDisulfideBonded(atom):
+            for b in self._bonds:
+                if (atom in b and b[0].name == 'SG' and
+                    b[1].name == 'SG'):
+                    return True
+
+            return False
 
         cyx = [res for res in self.residues() if res.name == 'CYS' and isCyx(res)]
         atomNames = [[atom.name for atom in res._atoms] for res in cyx]
         for i in range(len(cyx)):
             sg1 = cyx[i]._atoms[atomNames[i].index('SG')]
             pos1 = positions[sg1.index]
+            candidate_distance, candidate_atom = 0.3*nanometers, None
             for j in range(i):
                 sg2 = cyx[j]._atoms[atomNames[j].index('SG')]
                 pos2 = positions[sg2.index]
                 delta = [x-y for (x,y) in zip(pos1, pos2)]
                 distance = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2])
-                if distance < 0.3*nanometers:
-                    self.addBond(sg1, sg2)
+                if distance < candidate_distance and not isDisulfideBonded(sg2):
+                    candidate_distance = distance
+                    candidate_atom = sg2
+            # Assign bond to closest pair.
+            if candidate_atom:
+                self.addBond(sg1, candidate_atom)
+
+
 
 class Chain(object):
     """A Chain object represents a chain within a Topology."""
EOF
)

echo "Testing out Jax"
python -c "import jax; print(jax.local_devices()[0].platform)"

echo "conda env is setup, now run the script"

QUERY_FASTA=$1
QUERY_BASE=${QUERY_FASTA%%.*}

WORKDIR=work
mkdir -p ${WORKDIR}
cp "${QUERY_FASTA}" "${WORKDIR}"
cp run_alphafold.py "${WORKDIR}"
tar xzvf hmmer_output.tar.gz -C "${WORKDIR}"

mkdir alphafold_params

cd alphafold_params
(for letter in {a..d}; do echo $letter; done) | xargs -n1 -P2 bash -c \
        'i=$0; 
url="http://proxy.chtc.wisc.edu/SQUID/dcosta2/alphafold_params/alphafold_params_colab_tar_a${i}"; 
wget ${url}'
cd ..

mkdir -p alphafold/data/params
cat alphafold_params/* | tar xzvf - -C alphafold/data/params
rm -r alphafold_params/

cd "${WORKDIR}"
python run_alphafold.py
