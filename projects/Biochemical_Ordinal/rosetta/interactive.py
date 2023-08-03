import sys
import logging
logging.basicConfig(level=logging.INFO)
#import matplotlib
#import pandas as pd
import os
import pyrosetta
#from pyrosetta import *
#from pyrosetta.rosetta.protocols.rosetta_scripts import *
import pyrosetta.distributed.viewer as viewer
#from pyrosetta.rosetta.protocols.ligand_docking import *

USING_PYMOL = True

pdb_filename = "SadA_NSLeu_Corrected_3701_best_structure_0044.pdb" 
ligand_params = "NEU.params"
ligand_conformers = "NEU_conformers.pdb"
AKG_params = "AKG.params" 



flags = f"""
-extra_res_fa {AKG_params} {ligand_params}
-s '{pdb_filename}'
-mute all
-ex1
-ex2
-no_optH false
-flip_HNQ true
-ignore_ligand_chi true
-overwrite
-mistakes
"""
pyrosetta.distributed.init(flags)

pymover = None
if USING_PYMOL:
    pymover = pyrosetta.PyMOLMover()
    pymover.keep_history(True)

    

pose = pyrosetta.io.pose_from_file(filename=pdb_filename)

if USING_PYMOL: pymover.apply(pose)

pyrosetta.toolbox.mutants.mutate_residue(pose, 1, "A")

if USING_PYMOL: pymover.apply(pose)




    
