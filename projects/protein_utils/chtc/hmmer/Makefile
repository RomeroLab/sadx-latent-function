
clean:
	rm -f *.out *.err *.log

# make small fasta for testing
small_fasta: uniref100_small.fasta.gz
	zcat /staging/dcosta2/uniref100.fasta.gz | head -999 | gzip --best  > uniref100_small.fasta.gz

# submit small job 
ss: small_fasta
	condor_submit test_hmmer_small.sub
