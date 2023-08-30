#!/bin/bash
while :
do
  condor_release -all
  echo "Release run on : " `date`
  sleep 20m
done
