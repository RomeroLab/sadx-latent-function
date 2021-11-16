
clean:
	rm -f ROSETTA_CRASH.log
	rm -rf working
	rm -f inputs.tar.gz
	rm -f logs/*

move_results:
	mv -f SadA_*_rosetta.tar.gz results/

deepclean: clean
	rm -f *.tar.gz

inputs.tar.gz: *.pdb *.xml *.params
	tar zcf inputs.tar.gz *.pdb *.xml *.params

# test on group server
test_gs: inputs.tar.gz
	./ligand_dock.sh 0 SadA_NSLeu_Corrected_3701_0002.pdb "X42A.X100D" 100


# submit on chtc
submit_wt: inputs.tar.gz
	condor_submit \
			START_STRUCT=SadA_NSLeu_Corrected_3701_0002.pdb \
			VARIANT_LIST_FILE=variant_list_WT.txt \
			ligand_dock.sub

submit_2d: inputs.tar.gz
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_EPPCR.txt \
			ligand_dock.sub
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random.txt \
			ligand_dock.sub

