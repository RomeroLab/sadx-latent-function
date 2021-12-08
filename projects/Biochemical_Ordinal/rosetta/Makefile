RDRIVE_DIR=/mnt/rdrive/promero2/General/Sameer/SadA_azidation/rosetta
RDRIVE_TAR=$(RDRIVE_DIR)/2D_triple_mutants.tar
CHTC_RESULTS_PATH=sameerd/projects/Biochemical_Ordinal/rosetta/results
SCRATCH_OUTPUT_DIR=../data/scratch/rosetta/2D_triple_mutants

clean:
	rm -f ROSETTA_CRASH.log
	rm -rf working
	rm -f inputs.tar.gz
	rm -f logs/*

tidy:
	mv -f SadA_*_rosetta.tar.gz results/
	rm -f logs/*.err
	rm -f logs/*.out 

move_chtc_results:
	scp dcosta2@submit-1.chtc.wisc.edu:${CHTC_RESULTS_PATH}/"*" \
		${SCRATCH_OUTPUT_DIR} 
	ssh dcosta2@submit-1.chtc.wisc.edu 'rm ${CHTC_RESULTS_PATH}/*'

# CHTC results re-done on test set
move_chtc_test_set_results:
	mkdir -p ${SCRATCH_OUTPUT_DIR}_test_set
	scp dcosta2@submit-1.chtc.wisc.edu:${CHTC_RESULTS_PATH}/"*" \
		${SCRATCH_OUTPUT_DIR}_test_set
	ssh dcosta2@submit-1.chtc.wisc.edu 'rm ${CHTC_RESULTS_PATH}/*'

backup_results:
	mv -f ${RDRIVE_TAR} ${RDRIVE_TAR}.bak
	tar cf ${RDRIVE_TAR} -C ../data/scratch/rosetta 2D_triple_mutants
	du -s -h ${RDRIVE_TAR}

backup_test_set_results:
	tar cf ${RDRIVE_DIR}/2D_triple_mutants_test_set.tar \
    		   	-C ../data/scratch/rosetta 2D_triple_mutants_test_set
	du -s -h ${RDRIVE_DIR}/2D_triple_mutants_test_set.tar


deepclean: clean
	rm -f *.tar.gz

inputs.tar.gz: *.pdb *.xml *.params options_*.txt
	tar zcf inputs.tar.gz *.pdb *.xml *.params options_*.txt

# test on group server
test_gs: inputs.tar.gz
	./ligand_dock.sh 0 SadA_NSLeu_Corrected_3701_0002.pdb "X42A.X100D" 10

# test difference between xml output and example script from Mark
# should be identical and show no differences
test_variant_xml:
	bash -c 'diff <(./create_variant_xml.sh X65G.X71G.X157G A) attic/SadA_mutate_example.xml'


# submit on chtc
submit_test: inputs.tar.gz
	condor_submit \
			START_STRUCT=SadA_NSLeu_Corrected_3701_0002.pdb \
			VARIANT_LIST_FILE=variant_list_test.txt \
			ligand_dock.sub

submit_wt: inputs.tar.gz
	condor_submit \
			START_STRUCT=SadA_NSLeu_Corrected_3701_0002.pdb \
			VARIANT_LIST_FILE=variant_list_WT.txt \
			ligand_dock.sub

submit_2D_random: inputs.tar.gz
	#@for f in $(shell ls variant_list_2D_random_a[a-e].txt); do \
	#@for f in $(shell ls variant_list_2D_random_a[f-g].txt); do \
	#@for f in $(shell ls variant_list_2D_random_a[h-i].txt); do \
	#
	@for f in $(shell ls variant_list_2D_random_a[j-l].txt); do \
		condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=$${f} \
			ligand_dock.sub; \
	 done


submit_2D_test_set: inputs.tar.gz
	@for f in $(shell ls variant_list_2D_test_set_a[a-b].txt); do \
		condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=$${f} \
			ligand_dock.sub; \
	 done




