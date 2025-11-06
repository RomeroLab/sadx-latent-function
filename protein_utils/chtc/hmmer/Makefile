
clean:
	@rm -rf output
	@mkdir -p output

prepare_files: 
	@mkdir -p output
	@echo "Found" `ls -al /staging/dcosta2/hmmer_* 2> /dev/null | wc -l` \
				"output files which may be overwritten"

# submit small job 
ss: prepare_files
	condor_submit test_hmmer_small.sub

jackhmmer: prepare_files
	condor_submit hmmer_full.sub

# fetching sequences from uniref100 file
sfetch_small: prepare_files
	condor_submit test_sfetch_small.sub

sfetch: prepare_files
	condor_submit sfetch_full.sub

