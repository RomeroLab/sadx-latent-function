# SadX Combinatorial Library

This repository provides an open-source implementation for generating and prioritizing SadX enzyme variants using the CORAL loss function. Its main purpose is to document the training process of the variants described in the SadX paper.
Please note that this library is not optimized for use with other proteins.

## Setup 

## Training a Model with CORAL Loss for SadX 


## Running Inference 

(predictions from trained models)


## Log 

1) Migrated the whoe repository from sameerd to here. Added in relevant files for training on SadX to `output` file.
The rest is reletively constant. 

2) Downloaded and untarrred directory from romero drive. So that MSA files and updates to python files are documented. 

3) Archived (updated November 6,2025),  to access these files again look to `sameerd`
* Rosetta Runs with RosettaLigand 
* Attic of Marks Rosetta Runs 
* ESM finetuning 
* Protein MPNN predictions (have many other predictors in METL paper)
* AlphaFold Predictions (on Database now)


## CCM todo
- CCM project [1-2 days] 
  - Finally get data from other lysate mutants 
  - Get weights for the predictions (1)
  - Figure out how to run DCA [1.5 hours] 
  - Figure out how inference is done one those variants (2) [
  - Find out how DCA predictions were done and then run DCA predictions on new tested variants 
  - Preprocess dms data 
  - Run inference on variants 
  - How did he make the prioritization file. I think Sameer will just have to be the one who answers that unless it’s literally just the rankings . 
  - Does Sameer want to include any sequencing stuff from oct 2022. Currently removed . He will have to add back in.
