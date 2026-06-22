# SadX Combinatorial Library

This repository provides an open-source implementation for generating and prioritizing SadX enzyme variants using the CORAL loss function. Its main purpose is to document the training process of the variants described in the SadX paper.
Please note that this library is not optimized for use with other proteins.


## Preprocessing: PacBio Sequencing → Training Data

The training data for the CORAL model comes from three rounds of directed evolution on SadX. Each round produced a library of ~900 variants made by error-prone PCR (epPCR):

| Library | Parent | Variants placed | Bins (H / P / L / N) |
|---------|--------|-----------------|----------------------|
| 1-VH | 1-VH.fasta | 809 | 4 / 2 / 349 / 454 |
| 2-L | 2-L.fasta | 900 | 29 / 52 / 532 / 287 |
| 3-VRL | 3-VRL.fasta | 899 | 81 / 222 / 266 / 330 |

Variants were screened in lysate for azidation activity and sorted into four ordinal bins: **H**igh, **P**arent-like, **L**ow, and **N**o activity. All three libraries were then pooled (with barcodes identifying each library×bin group) and sequenced on PacBio in a single CCS (Circular Consensus Sequencing) run. The preprocessing pipeline below converts the raw PacBio output into `data/sequences_Oct22.tsv`, which provides the sequence identity for each variant in each bin.

For annotated code walkthroughs of each step, see [`docs/preprocessing_code_reference.md`](docs/preprocessing_code_reference.md).

All scripts are run from the `scripts/` directory.

### Expected directory layout

```
data/
├── 1-VH.fasta                          # Parent reference sequences
├── 2-L.fasta
├── 3-VRL.fasta
├── SadA D157G.fasta                     # Original SadX wild-type (without MBP)
└── scratch/
    ├── r64120_20221013_213137/3_C01/    # Raw PacBio output
    │   ├── 00Samples                    # Barcode → sample mapping (tab-delimited)
    │   └── demux/                       # Demultiplexed BAMs (one per barcode)
    │       └── *.bam
    └── Oct22/                           # Working directory (created by pipeline)
```

The `demux/` directory contains the output of PacBio's `lima` demultiplexer, which sorts the pooled sequencing run back into 12 separate BAM files (one per barcode) based on the short synthetic DNA tags ligated to each library×bin group before pooling. The `00Samples` file maps each barcode to its sample name (e.g., `bc2055` → `1VH_H`).

### Step 1: Align reads to parent references

**Script:** `scripts/align_pacbio_reads_Oct22.sh`

```bash
conda activate base  # requires pbmm2
cd scripts/
./align_pacbio_reads_Oct22.sh | sh > "../data/scratch/Oct22/pbmm2_output.txt" 2>&1
```

For each of the 12 barcoded BAM files, this script aligns the CCS reads to the appropriate parent reference sequence using `pbmm2` (PacBio's minimap2 wrapper for CCS data). It determines which parent to use by reading the first character of the sample name from `00Samples` (e.g., `1VH_H` → `1` → align against `1-VH.fasta`). The script prints the `pbmm2` commands to stdout rather than executing them directly, which are then piped to `sh` — this lets you inspect the commands before running.

**Output:** 12 aligned BAM files in `data/scratch/Oct22/`, one per barcode. Each read now carries alignment coordinates describing how it maps to the parent sequence, encoded as a CIGAR string.

### Step 2: Filter reads and extract mutations

**Script:** `scripts/process_bams_Oct22.py`

```bash
cd scripts/
python process_bams_Oct22.py
```

This script opens each aligned BAM and applies three sequential filters to every read:

1. **Alignment structure.** The read's CIGAR string must have soft clips on both ends (confirming the read spans the full gene with flanking vector sequence) and only exact matches or substitutions in the interior. Any read with insertions or deletions is discarded.
2. **Length.** The aligned region between the soft clips must be exactly 822 bp (positions 53–875), the expected length of the gene.
3. **Base quality.** Every base in the aligned region must have a Phred quality score ≥ 93 (~1 in 2 billion error probability). This extreme threshold is achievable with CCS reads because each molecule is sequenced multiple times in a circle, and the consensus across passes drives the error rate down.

For reads that survive, mutations are encoded relative to the parent in a compact shorthand: `G43T,T454C` means position 43 changed from G→T and position 454 from T→C. Reads identical to the parent are encoded as `*0*`.

**Output:** One TSV per barcode in `data/scratch/Oct22/` with columns `ss_dna`, `dna_dist`, `ss_aa`, `aa_dist`, `ref_dist`. Also writes `data/scratch/Oct22/Samples.csv` tracking how many reads survived each filter stage.

### Step 3: Aggregate, filter, and produce final dataset

**Notebook:** `scripts/nb_preprocess_bam_Oct22.ipynb`

This notebook reads the 12 per-barcode TSVs from step 2, then applies four additional filters to produce the final dataset.

**3a — Count duplicate reads.** Many reads share the same DNA sequence (the same variant sequenced thousands of times). The notebook groups identical sequences and counts occurrences. The resulting `dna_count` column is the confidence metric — a variant with 20,000 supporting reads is almost certainly real; one with 3 is likely noise.

**3b — Keep top N per bin.** Christian placed a known number of variants in each bin (e.g., 4 in 1VH-H, 532 in 2L-L). Only the top N sequences by read count are kept per bin, cutting the long tail of low-confidence sequences.

**3c — Read count cutoff.** Sequences below a minimum `dna_count` threshold are removed. The threshold is set by inspecting rank-abundance curves and identifying the steep dropoff separating real variants from noise. Most bins use a cutoff of 100; 1VH-H and 1VH-P use 10,000 (very few variants placed, lots of low-count junk); 2L-H uses 1,000.

**3d — Distance and stop codon filters.** Sequences with ≥ 12 DNA mutations from the parent are removed (too many for epPCR — likely chimeras or misreads). Sequences whose translation contains a premature stop codon are also removed.

**Output:** `data/sequences_Oct22.tsv` — 1,490 unique variants across the three libraries:

| Column | Description |
|--------|-------------|
| `ss_dna` | DNA mutations vs parent (e.g., `G43T,T454C,T594A`) or `*0*` for parent |
| `dna_count` | Number of independent reads supporting this sequence |
| `dna_dist` | Hamming distance from parent (DNA) |
| `aa_dist` | Hamming distance from parent (amino acid) |
| `ss_aa` | Amino acid mutations vs parent (e.g., `A15S,F152L,N198K`) |
| `early_stop` | Premature stop codon flag (always `False` after filtering) |
| `seq_rank` | Rank by read count within its bin |
| `parent` | Library: `1VH`, `2L`, or `3VRL` |
| `category` | Activity bin: `H`, `P`, `L`, or `N` |

### Known data quality issues

- **Parent contamination across bins.** The parent sequence (`*0*`) appears in bins where it should not be (e.g., it is the 4th-ranked sequence in 1VH-High with 11,321 reads and the top sequence in 2L-High with 6,378 reads). This is likely due to sample carryover during library preparation or demultiplexing errors.
- **Early stop codons in high-activity bins.** Some sequences in the 3-VRL High bin contained premature stop codons before filtering. Since truncated proteins should not be functional, this suggests barcode misassignment or chimeric reads in that bin.
- **1-VH library anomalies.** The 1VH library shows unusual per-position mutation frequency distributions and a very high read-count cutoff was needed (10,000) for the H and P bins, suggesting its library preparation may have been less uniform than 2L or 3VRL.

## Training a Model with CORAL Loss for SadX 


## Generating Paper Figures
 
All figure scripts output to timestamped directories under `output/model_evals_analysis/` so that each run is preserved with its config, log, raw data, and figure.
 
### Model Comparison (MAE Bar Chart + Configuration Table)
 
Compares all models on the held-out test set using Mean Absolute Error, alongside a table summarizing each model's configuration (read directly from the yaml files). Includes two constant baselines (Always Dead, Always Low).
 
```bash
python eval_mae.py eval_config.yml
```
 
Output directory: `output/model_evals_analysis/model_evals_<timestamp>/`
- `mae_comparison.png` — two-panel figure: configuration table (left) and MAE bar chart (right)
- `mae_results.csv` — raw MAE values and model properties
- `eval_config.yml` — copy of the config used
- `eval_mae.log` — full log
 
### DCA Score Box Plots
 
Faceted box plots showing the distribution of bmDCA density scores across activity categories (Dead, Low, Parent, High) for each parent library. Spearman correlation between DCA score and ordinal category is shown in each subplot title.
 
```bash
python plot_dca_boxplots.py output/ordinal_Oct22_sequences_with_dca_score.csv
```
 
Output directory: `output/model_evals_analysis/dca_<timestamp>/`
- `dca_score_boxplots.png` — faceted box plot (one panel per parent library)
- `dca_score_summary.csv` — summary statistics (count, mean, median, std, min, max) per parent and category
- `plot_dca_boxplots.log` — full log

## Setup 

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