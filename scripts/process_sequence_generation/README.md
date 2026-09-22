## Preprocessing: PacBio Sequencing → Training Data


Intermediate versions of the data — demultiplexed, quality-filtered reads from each processing step — are saved in the repository. This allows one to run the downstream analysis without having to reprocess from raw reads.

**We recommend skipping to Step 3. Steps 1 and 2 handle quality filtering of raw PacBio reads. Step 3 involves visualization of the final dataset with interpretable quality filters and uses pre-computed results from Step 2.**

All scripts are run from the `scripts/` directory.

### Step 1: Align reads to parent references (skip recommended)

> We recommend starting from Step 3, as Steps 1 and 2 require downloading and processing large raw sequencing files and have not been retested end-to-end. 

Raw sequences can be downloaded from the SRA (PacBio circular consensus sequencing reads have been deposited in the NCBI Sequence Read Archive under accession [PRJNA1505729](https://www.ncbi.nlm.nih.gov/sra/PRJNA1505729)).

Once downloaded put the reads into the expected directory layout below to run the preprocessing scripts. 

Expected directory layout:

```
data/
├── 1-VH.fasta                          # Parent reference sequences
├── 2-L.fasta
├── 3-VRL.fasta
├── SadA D157G.fasta                     # Original SadX wild-type (without MBP)
└── scratch/
    ├── r64120_20221013_213137/3_C01/    # Raw PacBio output
        ├── 00Samples                    # Barcode → sample mapping (tab-delimited)
        └── demux/                       # Demultiplexed BAMs (one per barcode)
            └── *.bam
```

The `demux/` directory contains the output of PacBio's `lima` demultiplexer, which sorts the pooled sequencing run back into 12 separate BAM files (one per barcode) based on the short synthetic DNA tags ligated to each library×bin group before pooling. The `00Samples` file maps each barcode to its sample name (e.g., `bc2055` → `1VH_H`).

> Note: The file 00Samples was lost, as such we regenerated it using the following [script](../regenerate_00Samples.py) using `data/Oct22/Samples.csv`.

**Script:** [`scripts/align_pacbio_reads_Oct22.sh`](../align_pacbio_reads_Oct22.sh)

```bash
conda create -n pbmm2_env -c bioconda pbmm2
conda activate pbmm2_env
cd scripts/
./align_pacbio_reads_Oct22.sh | sh > "../data/scratch/Oct22/pbmm2_output.txt" 2>&1
```

For each of the 12 barcoded BAM files, this script aligns the CCS reads to the appropriate parent reference sequence using `pbmm2` (PacBio's minimap2 wrapper for CCS data). It determines which parent to use by reading the first character of the sample name from `00Samples` (e.g., `1VH_H` → `1` → align against `1-VH.fasta`). The script prints the `pbmm2` commands to stdout rather than executing them directly, which are then piped to `sh` — this lets you inspect the commands before running.

**Output:** 12 aligned BAM files in `data/scratch/Oct22/`, one per barcode. Each read now carries alignment coordinates describing how it maps to the parent sequence, encoded as a CIGAR string.

### Step 2: Filter reads and extract mutations (skip recommendeded)

The pre-computed bam files from Step-1 are located in [data/Oct22](../../data/Oct22). 

In the script [`scripts/process_bams_Oct22.py`](../process_bams_Oct22.py), you will need to replace:
`../data/scratch/Oct22` with `../data/Oct22`.

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

We have saved a copy of the outputs from step in [data/Oct22](../../data/Oct22). 

### Step 3: Aggregate, filter, and produce final dataset


**Notebook:** [`scripts/nb_preprocess_bam_Oct22.ipynb`](../nb_preprocess_bam_Oct22.ipynb)

This notebook reads the 12 per-barcode TSVs from step 2, then applies four additional filters to produce the final dataset.

**3a — Count duplicate reads.** Many reads share the same DNA sequence (the same variant sequenced thousands of times). The notebook groups identical sequences and counts occurrences. The resulting `dna_count` column is the confidence metric — a variant with 20,000 supporting reads is almost certainly real; one with 3 is likely noise.

**3b — Keep top N per bin.** Christian placed a known number of variants in each bin (e.g., 4 in 1VH-H, 532 in 2L-L). Only the top N sequences by read count are kept per bin, cutting the long tail of low-confidence sequences.

**3c — Read count cutoff.** Sequences below a minimum `dna_count` threshold are removed. The threshold is set by inspecting rank-abundance curves and identifying the steep dropoff separating real variants from noise. Most bins use a cutoff of 100; 1VH-H and 1VH-P use 10,000 (very few variants placed, lots of low-count junk); 2L-H uses 1,000.

**3d — Distance and stop codon filters.** Sequences with ≥ 12 DNA mutations from the parent are removed (too many for epPCR — likely chimeras or misreads). Sequences whose translation contains a premature stop codon are also removed.

**Output:** [`data/sequences_Oct22.tsv`](../../data/sequences_Oct22.tsv) — 1,490 unique variants across the three libraries:

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


### Step 4: Final Dataset Processing

The final processing step took place in  [`nb_natural_seqs_pssm_dca.ipynb`](../nb_natural_seqs_pssm_dca.ipynb). Look to section `check correlation with 1-VH, 2-L, 3-VRL libraries`. 
However, that noteboook contains lots of excess code for computing the DCA score (see section  see [DCA score README.md](../../scripts/bmDCA_sadA/README.md)
) and datasets which have since been discarded. It also calls code which is outside of this directory. 
To just generate the final 1,326 variants simply run the below commands in a python script. 

This code removes all parent amino acid duplicates across rounds, and replaces them with three parent sequences, which are placed in the parent bin, 1 per each library. 

>Note: this produces the final dataset `ordinal_Oct22_sequences_without_dca_score.csv`. The key here is **without_dca**. If you wish to add your own DCA scores look to [DCA score README.md](../../scripts/bmDCA_sadA/README.md). 
> The recommendation however simply to use the pre-generated dataset [output/ordinal_Oct22_sequences_without_dca_score.csv](../../output/ordinal_Oct22_sequences_without_dca_score.csv). 

```angular2html
# run from /scripts
seqs = pd.read_csv("../data/sequences_Oct22.tsv", sep="\t")
seqs_no_dups = seqs.drop_duplicates(["parent", "category", "ss_aa"], keep='first')
parent_seqs = seqs_no_dups[seqs_no_dups.ss_aa == "*0*"].groupby("parent").head(1)
parent_seqs.category = "P"
final_seqs = pd.concat([seqs_no_dups[seqs_no_dups.ss_aa != "*0*" ], parent_seqs])
final_seqs["response"] = final_seqs.category.map({"H":3, "P":2, "L":1, "N":0})

# map the shortened sequences to full sequences
final_seqs["sequence_aa"] = final_seqs[["ss_aa", "parent"]].apply(lambda x: 
                                    utils.expand_mut_str_list_to_seq(
                                        x['ss_aa'], parent_seqs[x['parent']], 
                                        split_mut_char=",", offset=1), axis=1)

final_seqs["sequence_aa_trim"] = final_seqs.sequence_aa.str.slice(1, -1)


(final_seqs[["parent", "category", "response", "sequence_aa_trim"]]
     .to_csv("../output/ordinal_Oct22_sequences_without_dca_score.csv", index=False))

```

#### Known data quality issues

- **Parent contamination across bins.** The parent sequence (`*0*`) appears in bins where it should not be (e.g., it is the 4th-ranked sequence in 1VH-High with 11,321 reads and the top sequence in 2L-High with 6,378 reads). This is likely due to sample carryover during library preparation or demultiplexing errors.
- **Early stop codons in high-activity bins.** Some sequences in the 3-VRL High bin contained premature stop codons before filtering. Since truncated proteins should not be functional, this suggests barcode misassignment or chimeric reads in that bin.
- **1-VH library anomalies.** The 1VH library shows unusual per-position mutation frequency distributions and a very high read-count cutoff was needed (10,000) for the H and P bins, suggesting its library preparation may have been less uniform than 2L or 3VRL.


### Part 5: Figures in Manuscript

All of the figures were generated via a copy of the [scripts/nb_preprocess_bam_Oct22.ipynb](../nb_preprocess_bam_Oct22.ipynb), [nb_preprocess_bam_Oct22_FIGURES_ONLY.ipynb](nb_preprocess_bam_Oct22_FIGURES_ONLY.ipynb). 
This copy was used only to generate figures for the manuscript. This was done to guarantee the [scripts/nb_preprocess_bam_Oct22.ipynb](../nb_preprocess_bam_Oct22.ipynb) notebook was not manipulated and there was a documented version of how the data was processed.

The output figures from this ipynb are in [output/process_seq_data](../../output/process_seq_data). 


