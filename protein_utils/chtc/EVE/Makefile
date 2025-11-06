
.PHONY: results

fake:
	ls

clean:
	rm -f logs/*

deepclean: clean


test:
	condor_submit TEST=1 pytorch_gpu.sub

testcpu:
	condor_submit TEST=1 NOGPU=1 pytorch_gpu.sub

submit:
	condor_submit pytorch_gpu.sub

