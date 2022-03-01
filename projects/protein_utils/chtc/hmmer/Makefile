
clean:
	@rm -rf output
	@mkdir -p output

prepare_files: 
	@mkdir -p output
	@echo "Found" `ls -al /staging/dcosta2/hmmer_* | wc -l` \
				"output files which will be overwritten"

# submit small job 
ss: prepare_files
	condor_submit test_hmmer_small.sub

submit: prepare_files
	condor_submit hmmer_full.sub
