
clean:
	@rm -rf output
	@mkdir -p output

deepclean: clean
	rm -f uniref100_small.fasta.gz

uniref100_small.fasta.gz:
	zcat /staging/dcosta2/uniref100.fasta.gz | head -999 | gzip --best  > uniref100_small.fasta.gz

# make small fasta for testing
prepare_files: uniref100_small.fasta.gz
	@mkdir -p output

# submit small job 
ss: prepare_files
	mkdir -p output
	@echo "Found" `ls -al /staging/dcosta2/hmmer_* | wc -l` \
				"output files which will be overwritten"
	condor_submit test_hmmer_small.sub
