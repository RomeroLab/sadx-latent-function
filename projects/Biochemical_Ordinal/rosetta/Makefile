
clean:
	rm -f ROSETTA_CRASH.log
	rm -f score.sc
	rm -rf working
	rm -f inputs.tar.gz
	rm -f logs/*

inputs.tar.gz: *.pdb *.xml *.params options_*.txt 
	tar zcf inputs.tar.gz *.pdb *.xml *.params options_*.txt 

# test on group server
test_gs: inputs.tar.gz
	./ligand_dock.sh 0 "X42A.X100D" 100


# submit on chtc
submit: inputs.tar.gz
	condor_submit ligand_dock.sub

