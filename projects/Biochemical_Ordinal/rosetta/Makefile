
noop:
	@echo

clean:
	rm -f ROSETTA_CRASH.log
	rm -rf working
	rm -f inputs.tar.gz
	rm -f logs/*
	rm -f SadA_X*_rosetta.tar.gz


inputs.tar.gz: *.pdb *.xml *.params options_*.txt
		tar zcf inputs.tar.gz *.pdb *.xml *.params options_*.txt

test_gs: inputs.tar.gz
		./ligand_dock.sh 0  SadA_NSLeu_Corrected_3701_best_structure_0044.pdb "X1A" 10


single_mutants:
	@python generate_variants.py \
			-p SadA_NSLeu_Corrected_3701_best_structure_0044.pdb \
				> SadA_NSLeu_Corrected_3701_best_structure_0044.single_mutants.txt
