

pass:

clean:
	rm -f logs/*

deepclean: clean
	rm -f hmmer_output/*
	rm -f hmmer_output.tar.gz

sweep:
	mv *.tblout.txt *.sto hmmer_output/
	tar zcf hmmer_output.tar.gz hmmer_output

test_hmmer:
	condor_submit hmmer.sub

inter_hmmer:
	condor_submit -i hmmer.sub

inter_cpu_alphafold:
	condor_submit DISABLE_GPU=1 -i alphafold.sub

cpu_alphafold:
	condor_submit DISABLE_GPU=1 alphafold.sub

inter_alphafold:
	condor_submit -i alphafold.sub

alphafold:
	condor_submit alphafold.sub

hmmer:
	condor_submit hmmer.sub

