
submit: code
	condor_submit esm.sub

code: code.tar.gz
	tar zcf code.tar.gz -C modified_code --dereference .

clean:
	rm -f code.tar.gz
	rm -f run_output/condor_logs/*

inter: code
	condor_submit -i esm.sub 
