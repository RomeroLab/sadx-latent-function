from subprocess import Popen,STDOUT,PIPE
from os import remove
import os


threetoone={'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
            'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V'}
onetothree = dict([(threetoone[x],x) for x in threetoone.keys()])





mutate_template = open('SadA_mutate.xml').read().strip()

for pos in range(1,274):
    for aa, one_letter in threetoone.items(): 

        # write mut.xml file 
	
	
        mut_file = mutate_template.replace(str("POSITION"),str(pos)).replace(str("AMINOACID"),aa)
        open('tmp_mut_file.xml','w').write(mut_file)

        # run mutate command
        cmd = '/mnt/scratch/software/rosetta/rosetta.binary.linux.release-296/main/source/bin/rosetta_scripts.static.linuxgccrelease @options_mutate.txt '
        Popen(cmd,shell=True).wait()

        if os.path.isfile('Mutant_directories/Mutants1/SadA_NSLeu_Corrected_3701_best_structure_0044_0001.pdb')==True:
            os.rename('Mutant_directories/Mutants1/SadA_NSLeu_Corrected_3701_best_structure_0044_0001.pdb', 'Mutant_directories/Mutants1/temp.pdb')


        terms=[]
        # run dock command
        cmd = '/mnt/scratch/software/rosetta/rosetta.binary.linux.release-296/main/source/bin/rosetta_scripts.static.linuxgccrelease @options_dock.txt -out:suffix '
        Popen(cmd,shell=True).wait()


        

        if os.path.isfile('Output/score.sc')==True:	
            new_score=open('Output/score.sc').read().strip().replace(str('temp_0001'),'SadA_NSLeu_'+str(pos)+aa+'.pdb'+'\n')

            open('Output/score.sc','w').write(new_score)

            os.rename('Output/temp_0001.pdb','Output/SadA_NSLeu_'+str(pos)+aa+'.pdb')

        break
    break
