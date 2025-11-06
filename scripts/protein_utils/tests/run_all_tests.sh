#!/bin/bash

for test_filename in test_*.py; do
  echo ">>> Running : ${test_filename}"
  PYTHONPATH="../../" python ${test_filename}
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> "
done 


