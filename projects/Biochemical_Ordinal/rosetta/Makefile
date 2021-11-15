
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
	./ligand_dock.sh 0 "X42A.X100D" 100


# submit on chtc
submit: inputs.tar.gz
	condor_submit ligand_dock.sub

