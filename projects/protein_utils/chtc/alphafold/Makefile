

pass:


clean:
	rm -f logs/*

deepclean: clean
	rm -f output/*

sweep:
	mv *.tblout.txt *.sto output/

test:
	condor_submit hmmer.sub

inter:
	condor_submit -i hmmer.sub

submit:
	condor_submit hmmer.sub

