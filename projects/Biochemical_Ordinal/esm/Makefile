
submit:
	condor_submit esm.sub

code:
	tar zcf code.tar.gz -C modified_code --dereference .

clean:
	rm -f run_output/condor_logs/*

inter:
	condor_submit -i esm.sub 
