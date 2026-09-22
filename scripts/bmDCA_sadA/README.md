# MSA Generation and Direct Coupling Analysis Pipeline

Build a filtered, reweighted multiple sequence alignment (MSA) from a query protein sequence, then fit a Potts model using Boltzmann machine direct coupling analysis (bmDCA) to learn evolutionary couplings.

## Prerequisites

- **HMMER 3.4** (includes `jackhmmer` and `esl-reformat`)
- **bmDCA** from [ranganathanlab/bmDCA](https://github.com/ranganathanlab/bmDCA) (includes `bmdca` and `arma2ascii`)
- **Python 3** with the following packages:
  - BioPython
  - NumPy
  - PyTorch
- **UniRef90** database (local copy compressed ~40GB, ~130GB uncompressed)

### Installing conda environment

## Environment Setup

Create the conda environment from the setup file:

```bash
conda env create -f setup/environment_minimal.yaml
conda activate sadA
```

### Installing HMMER

> Only install HMMER and do Part 1: MSA Generation if you wish to recurate the Multiple sequence alignment. Given the computational difficulty in doing this (downloading a large database and new software) it is recommended to use the already pre-saved [MSA](../../data/msa/sadA_full_clean.fasta).

```bash
wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz
tar zxf hmmer-3.4.tar.gz
cd hmmer-3.4/
./configure
make
sudo make install
cd easel
sudo make install
```

## Part 1: MSA Generation

>Note: For Part 1: MSA Generation, all steps were done in the [build_msa.sh](build_msa.sh), however these contain hardcoded paths to local files. The commands below describe how to obtain the same results using the scripts in this folder (`bmDCA_sadA`) and be able to run on your local machine if you wish to reproduce the results entirely.


All scripts are expected to be in the same directory from `bmDCA_sadA`. Set these environment variables for your run:

```bash
QUERY=path/to/query.fasta        # query protein sequence
DB=path/to/uniref90.fasta        # local UniRef90 database
STUB=sadA_full                    # prefix for all output files
```


### Step 1: Homology search

Iteratively search the query against UniRef90 with jackhmmer. This builds a profile HMM from the query, searches the database, incorporates significant hits, and repeats until convergence. All default parameters were used.

```bash
jackhmmer -A ${STUB}.sto -o ${STUB}.out.txt ${QUERY} ${DB}
```

### Step 2: Format conversion

Convert the Stockholm alignment to aligned FASTA. The `-u` flag strips annotation markup.

```bash
esl-reformat -u -o ${STUB}.afa afa ${STUB}.sto
```

### Step 3: Alignment filtering

Clean the raw MSA. This script does four things in order:

1. Trims columns to only positions where the query has a residue (non-gap), so every column maps to a query position.
2. Removes sequences with fewer non-gap residues than 50% of the query length.
3. Removes sequences with any amino acid repeated more than 10 consecutive times.
4. Removes sequences containing non-canonical amino acids.

```bash
python3 jackhmmer_aligned_msa_filter.py \
    -i ${STUB}.afa \
    -q ${QUERY} \
    -o ${STUB}_clean.fasta
```

Optional flags: `-f 0.5` (length fraction cutoff, default 0.5), `-r 10` (repeat cutoff, default 10).

### Step 4: Sequence reweighting

Compute phylogenetic weights to correct for uneven sampling across the tree of life. For each sequence, counts how many neighbors share ≥80% sequence identity, then assigns weight = 1 / neighbor_count. Weights are normalized across homologs. Uses GPU when available.

```bash
python3 reweighting_tools.py \
    -i ${STUB}_clean.fasta \
    -o ${STUB}_clean.weights.npy
```

Optional flags: `-t 0.8` (identity threshold, default 0.8), `-d cuda:0` (device, default auto-detect).

### Outputs

- `${STUB}_clean.fasta` — filtered, query-anchored MSA
- `${STUB}_clean.weights.npy` — per-sequence weights (NumPy array, length N)

Intermediate files (`*.sto`, `*.out.txt`, `*.afa`) can be deleted after the pipeline completes.

> The precomputed versions of the [MSA](../../data/msa/sadA_full_clean.weights.npy) and [weights](../../data/msa/sadA_full_clean.weights.npy). 


## Part 2: Direct Coupling Analysis (bmDCA)

Fit a Potts model to the filtered MSA using Boltzmann machine learning. This learns two sets of parameters: **fields** h(i, a) capturing the independent preference for amino acid *a* at position *i*, and **couplings** J(i, j, a, b) capturing the pairwise interaction between amino acid *a* at position *i* and amino acid *b* at position *j*.

> Steps 5-6 of this were run via the [`run.sh`](run.sh) script. But as stated above the below commands do not contain hardcoded paths as this file does. 

### Step 5: Run bmDCA

> There is no saved version of the DCA parameters. Only two options are to compute them below, or use the precomputed [DCA scores](../../output/ordinal_Oct22_sequences_with_dca_score.csv).

```bash
mkdir -p run

bmdca \
    -i ${STUB}_clean.fasta \
    -r -d ./run \
    -c bmdca.conf
```

The `-r` flag enables sequence reweighting. The `-d` flag sets the output directory. Parameters are saved in Armadillo binary format every 20 iterations (set by `save_parameters` in `bmdca.conf`).

### Step 6: Convert parameters to text

Pick the iteration to use (here, 200) and convert from Armadillo binary to a text file:

```bash
ITER=200

arma2ascii \
    -p run/parameters_h_${ITER}.bin \
    -P run/parameters_J_${ITER}.bin
```

This produces `run/parameters_${ITER}.txt` with lines formatted as `J i j a b value` and `h i a value`.


### Step 7: Convert parameters to NumPy

```bash
python3 bmDCA.py \
    -i run/parameters_${ITER}.txt \
    -o ${STUB}
```

This outputs the learned parameters as NumPy arrays.

### Outputs

- `${STUB}_e_i_a_j_b.npy` — pairwise couplings, shape (L, q, L, q)
- `${STUB}_h_i_a.npy` — single-site fields, shape (L, q)

where L is the protein length and q = 21 (20 amino acids + gap).

### Part 3: Inference to Compute DCA for Directed Evolution libraries


Scores were computed in  [`nb_natural_seqs_pssm_dca.ipynb`](../nb_natural_seqs_pssm_dca.ipynb). Look to section `check correlation with 1-VH, 2-L, 3-VRL libraries`. 
However, that noteboook contains lots of excess code for sequence processing and datasets which have since been discarded. It also calls code which is outside of this directory. 
To just generate the scores from the learned parameters simply run the below commands in a python script. 

```

from encoding import DEFAULT_ENCODER
from energy.potts import energy_calc_single
import pandas as pd

bmDCA_params_dir = "/path/to/bmDCA_scores"
e_i_a_j_b = np.load(f"{bmDCA_params_dir}/sadA_e_i_a_j_b.npy")
h_i_a = np.load(f"{bmDCA_params_dir}/sadA_h_i_a.npy")

final_seqs= pd.read_csv("../../output/ordinal_Oct22_sequences_with_dca_score.csv")

final_seqs["dca_score_regenerated"] = final_seqs.sequence_aa_trim.map(lambda x:
    energy_calc_single(DEFAULT_ENCODER.string_to_np(x), h_i_a=h_i_a, e_i_a_j_b=e_i_a_j_b))

final_seqs.to_csv("../../output/ordinal_Oct22_sequences_with_dca_score_regenerated.csv",index=False)

```

### DCA figures

> Note you need to change to `ordinal_Oct22_sequences_with_dca_score_regenerated.csv` below and change the column name above to `dca_score` if you want to do the regenerated DCA scores in the visualization. 


**DCA Score Box Plots** — Faceted box plots showing bmDCA density score distributions across activity categories, with Spearman correlation per parent library.

To make the box plots run this command from the root directory:
```angular2html
python scripts/plot_dca_boxplots.py output/ordinal_Oct22_sequences_with_dca_score.csv
```

Output is given [here](../../output/model_evals_analysis/dca_20260922_123452/dca_score_boxplots.png). 

**MSA Visualization** — Visualize the sequence logo of the MSA. Outputs to [`output/msa_visualization/`](../../output/msa_visualization/msa_logo.png).

Run from root directory of repo.

```angular2html
python scripts/visualize_msa.py
```

## File Descriptions


### MSA generation

| File | Description |
|------|-------------|
| `jackhmmer_aligned_msa_filter.py` | Filters and trims the raw jackhmmer alignment |
| `reweighting_tools.py` | CLI wrapper for sequence reweighting |
| `reweighting.py` | Core reweighting algorithm (GPU-accelerated pairwise identity) |
| `dataloader.py` | MSA file I/O; reads FASTA/ALN into NumPy arrays or PyTorch datasets |
| `read_config.py` | YAML config reader and device selection (shared with the VAE/DCA codebase) |
| `build_msa.sh` | Original pipeline script with hardcoded paths from the SadA run (kept for reference) |

### bmDCA

| File | Description |
|------|-------------|
| `bmdca.conf` | bmDCA configuration file with all hyperparameters |
| `convert_bmDCA.sh` | Original wrapper that called `bmDCA.py` via `protein_utils` (kept for reference) |
| `bmDCA.py` | Reads bmDCA text parameters, re-encodes amino acids, saves as NumPy |
| `encoding.py` | Amino acid alphabet encoder used by `bmDCA.py` (from `protein_utils`) |
| `fileio.py` | File handle utilities for gzipped/plain text (from `protein_utils`) |
| `run.sh` | Original bmDCA run script with hardcoded paths from the SadA run (kept for reference) |
| `save_latest_params.sh` | Utility to extract parameters from the latest iteration |
| `list_intermediate_files.sh` | Utility to list intermediate parameter files for cleanup |