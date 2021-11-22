
clean:
	rm -f ROSETTA_CRASH.log
	rm -rf working
	rm -f inputs.tar.gz
	rm -f logs/*

move_results:
	mv -f SadA_*_rosetta.tar.gz results/

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

#submit_2d: inputs.tar.gz
	#condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_EPPCR.txt \
			ligand_dock.sub
	#condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random.txt \
			ligand_dock.sub

submit_2d_random: inputs.tar.gz
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random_aa.txt \
			ligand_dock.sub
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random_ab.txt \
			ligand_dock.sub
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random_ac.txt \
			ligand_dock.sub
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random_ad.txt \
			ligand_dock.sub
	condor_submit \
			START_STRUCT=SadA_NSLeu_2D.pdb  \
			VARIANT_LIST_FILE=variant_list_2D_random_ae.txt \
			ligand_dock.sub
