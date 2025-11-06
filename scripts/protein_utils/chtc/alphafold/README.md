# Running Alphafold2 on CHTC

This directory is an implementation of [Alphafold2 colab
v2.3.2](https://colab.research.google.com/github/deepmind/alphafold/blob/main/notebooks/AlphaFold.ipynb)
but running at CHTC. *Alphafold2 Colab* is an official stripped down version of
Alphafold2 that uses a smaller dataset to search for homologs and doesn't use
templates. Deepmind has tested it and it seems to have comparable accuracy as
the full Alphafold2 for most monomers. 

### Before using this implementation

* See if it is possible to download a structure from the [Alphafold Structure Database](https://alphafold.ebi.ac.uk/)
* See if it is feasible to use *Alphafold2 Colab* on Google colab
* See if it is possible to use *Colabfold* (an community version) of *Alphafold2 Colab* on Google colab.

If none of the above work, then running Alphafold2 Colab on CHTC might be a
decent alternative. 


### Overview
There are two steps. 
1. Use `hmmer` to search the various databases.
   ([Uniref90](https://www.uniprot.org/help/uniref), [Small
bfd](https://bfd.mmseqs.com/) soil metagenome clusters,
[Mgnify](https://www.ebi.ac.uk/metagenomics/) microbiome clusters). These
stripped down databases are around 300GB in total and are searched in chunks of
1GB in parallel.  It is quite fast since nearly all the jobs are run in
parallel at the same time and each job only uses a single cpu.
2. The results of the `hmmer` search are gathered up and used as input to the
   Alphafold script which runs on a gpu. 

### Detailed steps
1. Make a copy of this directory (incuding subdirectories) on CHTC somewhere.
2. Make a copy of `hmmer.tar.gz` in the location `../hmmer/hmmer.tar.gz`. This file can be copied from `/home/dcosta2/sameerd/projects/protein_utils/chtc/hmmer/hmmer.tar.gz` or downloaded from the research drive at `//research.drive.wisc.edu/promero2/General/Sameer/chtc/alphafold_executables/hmmer.tar.gz`. If it isn't possible to find this file, it can also be built quite easily using the [instructions in the hmmer directory](https://github.com/RomeroLab/sameerd/blob/master/projects/protein_utils/chtc/hmmer/README.md).
3. Edit `target.fasta` to replace the current sequence with your sequence.
4. Run `make deepclean` to make sure everything is ready to start
5. Run `make hmmer` to run JackHmmer against all the database chunks. It will spawn 199 jobs and they should all finish in a few minutes.
6. Run `make sweep` to compress all the results and get them ready for the next step
7. Run `make alphafold` to run alphafold. (Use `make cpu_alphafold` if you want to use the cpu instead of the GPU)
8. The result is a file called `prediction.tar.gz` that has a pdb for the best structure out of the 5 or 6 alphafold models that are run. The b-factors of the pdb file have the confidence values.



### Multimers
*Alphafold2 Colab* can be used for multimers as well but Deepmind recommends
using the full Alphafold2 for multimers due to the drop in accuracy. This
version of *Alphafold2 Colab* running on CHTC does not yet support multimers
but it should be possible to add with a little bit more work.


### TODO (pull requests welcome!)
* The relax functionality that uses *openmm* to resolve steric clashes needs to be tested
* Add multimer support
  * Multimer sequences go in `target.fasta`
  * `hmmer` needs to do each database chunks against all target sequences.
  * python class `PreCachedJackhmmer` needs to be modified to read each target's results
  * multimer python switches in [run_alphafold.py](run_alphafold.py) should be renabled

