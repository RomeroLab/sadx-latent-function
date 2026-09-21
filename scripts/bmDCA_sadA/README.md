# MSA Generation and Direct Coupling Analysis Pipeline

Build a filtered, reweighted multiple sequence alignment (MSA) from a query protein sequence, then fit a Potts model using Boltzmann machine direct coupling analysis (bmDCA) to learn evolutionary couplings.

## Prerequisites

- **HMMER 3.4** (includes `jackhmmer` and `esl-reformat`)
- **bmDCA** from [ranganathanlab/bmDCA](https://github.com/ranganathanlab/bmDCA) (includes `bmdca` and `arma2ascii`)
- **Python 3** with the following packages:
  - BioPython
  - NumPy
  - PyTorch
- **UniRef90** database (local copy)

### Installing HMMER

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

All scripts are expected to be in the same directory. Set these variables for your run:

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

## Part 2: Direct Coupling Analysis (bmDCA)

Fit a Potts model to the filtered MSA using Boltzmann machine learning. This learns two sets of parameters: **fields** h(i, a) capturing the independent preference for amino acid *a* at position *i*, and **couplings** J(i, j, a, b) capturing the pairwise interaction between amino acid *a* at position *i* and amino acid *b* at position *j*.

### Step 5: Run bmDCA

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




### DCA figures

**DCA Score Box Plots** — Faceted box plots showing bmDCA density score distributions across activity categories, with Spearman correlation per parent library.

**MSA Visualization** — Outputs to `output/msa_visualization/`.


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