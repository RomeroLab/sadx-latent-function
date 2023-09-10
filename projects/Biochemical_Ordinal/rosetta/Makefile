
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


inputs.tar.gz: *.pdb *.xml *.params options_*.txt create_variant_xml.sh
		cp prepare_parents/1VH.pdb .
		cp prepare_parents/2L.pdb .
		cp prepare_parents/3VRL.pdb .
		tar zcf inputs.tar.gz \
				*.pdb *.xml *.params options_*.txt create_variant_xml.sh
		rm -f 1VH.pdb 2L.pdb 3VRL.pdb

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
		NSTRUCT=40 \
		ligand_dock.sub

submit_screening_test: inputs.tar.gz
	condor_submit \
		START_STRUCT=1VH.pdb \
		VARIANT_LIST_FILE=screening_data/test_1VH.txt \
		NSTRUCT=10 \
		ligand_dock.sub

submit_screening_data: inputs.tar.gz
	condor_submit \
		START_STRUCT=1VH.pdb \
		VARIANT_LIST_FILE=screening_data/ordinal_Oct22_sequences_variants_for_1VH.txt  \
		NSTRUCT=1000 \
		ligand_dock.sub
	condor_submit \
		START_STRUCT=2L.pdb \
		VARIANT_LIST_FILE=screening_data/ordinal_Oct22_sequences_variants_for_2L.txt \
		NSTRUCT=1000 \
		ligand_dock.sub
	condor_submit \
		START_STRUCT=3VRL.pdb \
		VARIANT_LIST_FILE=screening_data/ordinal_Oct22_sequences_variants_for_3VRL.txt \
		NSTRUCT=1000 \
		ligand_dock.sub



