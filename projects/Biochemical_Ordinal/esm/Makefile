
submit: code.tar.gz
	condor_submit esm.sub

cpu: code.tar.gz
	condor_submit DISABLE_GPU=1 esm.sub

code.tar.gz:
	tar zcf code.tar.gz -C modified_code --dereference .

clean:
	rm -f code.tar.gz
	rm -f run_output/condor_logs/*

deepclean: clean
	rm -f training_logs.tar.gz

intercpu: code.tar.gz
	condor_submit DISABLE_GPU=1 -i esm.sub 

intergpu: code.tar.gz
	condor_submit -i esm.sub 
