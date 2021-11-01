

clean:
	rm -f ROSETTA_CRASH.log
	rm -f score.sc
	rm -r Models

# test on group server
test_gs:
	./ligand_dock.sh 0 options.txt 10


# submit on chtc
submit:
	condor_submit ligand_dock.sub

