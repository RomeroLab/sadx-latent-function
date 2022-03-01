
clean:
	@rm -rf output
	@mkdir -p output

prepare_files: 
	@mkdir -p output

# submit small job 
ss: prepare_files
	@echo "Found" `ls -al /staging/dcosta2/hmmer_* | wc -l` \
				"output files which will be overwritten"
	condor_submit test_hmmer_small.sub
