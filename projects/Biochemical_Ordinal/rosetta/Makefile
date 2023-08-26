
noop:
	@echo

clean:
	rm -f ROSETTA_CRASH.log
	rm -rf working
	rm -f inputs.tar.gz
	rm -f logs/*

deepclean: clean
	rm -f *.tar.gz

deepdeepclean: deepclean
	rm -f results/*

tidy:
	rm -f SadA_*_logs_rosetta.tar.gz
	mv -f SadA_*_rosetta.tar.gz results/
	rm -f logs/*.err
	rm -f logs/*.out 


inputs.tar.gz: *.pdb *.xml *.params options_*.txt
		tar zcf inputs.tar.gz *.pdb *.xml *.params options_*.txt

test_gs: inputs.tar.gz
		./ligand_dock.sh 0  SadA_NSLeu_Corrected_3701_best_structure_0044.pdb "X1A" 10

# submit on chtc
submit_test: inputs.tar.gz
	condor_submit \
		START_STRUCT=SadA_NSLeu_Corrected_3701_best_structure_0044.pdb \
		VARIANT_LIST_FILE=variant_list_test.txt \
		NSTRUCT=10 \
		ligand_dock.sub


single_mutants_andres:
	@python generate_variants.py \
			-p SadA_NSLeu_Corrected_3701_best_structure_0044.pdb \
				> SadA_NSLeu_Corrected_3701_best_structure_0044.single_mutants.txt

submit_andres: inputs.tar.gz
	condor_submit \
		START_STRUCT=SadA_NSLeu_Corrected_3701_best_structure_0044.pdb \
		VARIANT_LIST_FILE=SadA_NSLeu_Corrected_3701_best_structure_0044.single_mutants.txt\
		NSTRUCT=200 \
		ligand_dock.sub


single_mutants_alphafold:
	@python generate_variants.py \
			-p SadA_Alphafold_NSLeu_AKG_ZN.pdb \
				> SadA_Alphafold_NSLeu_AKG_ZN.single_mutants.txt

submit_test_alphafold: inputs.tar.gz
	condor_submit \
		START_STRUCT=SadA_Alphafold_NSLeu_AKG_ZN.pdb \
		VARIANT_LIST_FILE=variant_list_test.txt \
		NSTRUCT=10 \
		ligand_dock.sub


submit_alphafold: inputs.tar.gz
	condor_submit \
		START_STRUCT=SadA_Alphafold_NSLeu_AKG_ZN.pdb \
		VARIANT_LIST_FILE=SadA_Alphafold_NSLeu_AKG_ZN.single_mutants.txt\
		NSTRUCT=5000 \
		ligand_dock.sub


