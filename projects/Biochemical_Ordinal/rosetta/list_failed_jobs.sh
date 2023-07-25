#!/bin/bash

# look for jobs that don't have any output. (take 4.0K of space)
# jobs that work take up around 1M of space. 
# FIXME: Use filesize

ls -sk SadA_*.tar.gz | awk '{if ($1 < 10) print $2 }' | cut -d_ -f2
