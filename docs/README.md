# Preprocessing Code Reference

> Annotated code walkthrough for the PacBio → `sequences_Oct22.tsv` pipeline.
> For a summary of each step and how to run them, see the [README](../README.md#preprocessing-pacbio-sequencing--training-data).

## What you're starting with (the physical reality)

Christian has three libraries: **1-VH**, **2-L**, **3-VRL**. Each library contains ~900 protein variants made by error-prone PCR (epPCR). Through a screening assay in lysate, each variant was sorted into one of four activity bins:

| Bin code | Meaning | Example: 1VH placed |
|----------|---------|---------------------|
| H | High activity | 4 |
| P | Parent-like activity | 2 |
| L | Low activity | 349 |
| N | No activity | 454 |

These counts are hardcoded in `process_bams_Oct22.py`:

```python
# process_bams_Oct22.py, lines 58-69

num_samples_placed_d = {
        "1VH_H":4,
        "1VH_P":2,
        "1VH_L":349,
        "1VH_N":454,
        "2L_H":29,
        "2L_P":52,
        "2L_L":532,
        "2L_N":287,
        "3VRL_H":81,
        "3VRL_P":222,
        "3VRL_L":266,
        "3VRL_N":330
}
```

So there are **12 groups** total (3 libraries × 4 bins). Each group is a physically separate tube of DNA.

The problem: at this point, nobody knows what mutations are in these ~900 variants per library. Christian only identified ~10 by hand using Sanger sequencing. All 12 groups get sent to PacBio for next-generation sequencing to identify the rest.

---

## Before the pipeline: what PacBio does

### Barcoding and pooling

Before sending to PacBio, each of the 12 groups gets a short synthetic DNA tag ("barcode") ligated to both ends of every molecule. Group 1VH-H might get barcode `bc2055`, group 2L-N might get `bc2066`, etc. Then all 12 groups are pooled into a single tube for sequencing.

### Sequencing (CCS)

PacBio's sequencer reads individual DNA molecules. It reads each molecule multiple times in a circle (Circular Consensus Sequencing — CCS), then takes a consensus across those passes to produce one high-quality "read" per molecule. A **read** is: a DNA sequence + a per-base quality score.

### Demultiplexing

PacBio's software (`lima`) looks at the barcode on each end of each read and sorts reads back into their original 12 groups. Output: **12 demultiplexed BAM files**, one per barcode, stored in:

```
../data/scratch/r64120_20221013_213137/3_C01/demux/
```

A **BAM file** is a compressed binary format for storing sequencing reads. You can't open it in a text editor — you need a library like `pysam` (Python) to read it. Each read inside a BAM has: a DNA sequence, quality scores, and optionally alignment coordinates.

There's also a samples file that maps barcodes to groups:

```
../data/scratch/r64120_20221013_213137/3_C01/00Samples
```

This is a tab-delimited file with columns `Barcode` and `Bio_Sample` (e.g., `bc2055` → `1VH_H`).

---

## Stage 1: `align_pacbio_reads_Oct22.sh`

**Purpose:** Take each demultiplexed BAM and align every read to its parent reference sequence.

**What is alignment?** The reads from PacBio aren't just the gene — they include extra flanking DNA from the cloning vector, barcodes, etc. Alignment figures out: "positions 53-875 of this read correspond to the parent gene, and here's where it matches and where it differs." This is done by the tool `pbmm2` (PacBio's aligner for CCS reads).

**What is a FASTA file?** The simplest sequence format — just a name and a DNA sequence, no quality scores. The three parent references are stored as FASTA:

```
../data/1-VH.fasta
../data/2-L.fasta
../data/3-VRL.fasta
```

### The code

The script first defines a lookup table mapping library number → parent FASTA:

```bash
# align_pacbio_reads_Oct22.sh

declare -r -A parent_array=(
  [1]="${fasta_dir}/1-VH.fasta"
  [2]="${fasta_dir}/2-L.fasta"
  [3]="${fasta_dir}/3-VRL.fasta"
)
```

Then it reads the samples file line by line. For each barcode, it figures out which parent to align against by grabbing the first character of the sample name (e.g., `1VH_H` → `1` → use `1-VH.fasta`):

```bash
# align_pacbio_reads_Oct22.sh

sed '1,2d' "${samples_file}" | while read -r Barcode Parent || [ -n "$Barcode" ];
do
  echo "Processing : ${Barcode}"
  first_char=${Parent:0:1}
  input_filenames=(${demux_dir}/*${Barcode}--${Barcode}*.bam)
  # ...
  echo pbmm2 align \
          "${parent_array[${first_char}]}" \   # the reference FASTA
          "${input_filenames[0]}" \             # the demuxed BAM
          "${output_dir}/${Barcode}.bam" \      # the output aligned BAM
          "--preset CCS --log-level INFO --num-threads 8"
done
```

Note the `echo` — the script doesn't execute the commands directly. It *prints* them. You pipe the output into `sh` to actually run them:

```bash
(base) ./align_pacbio_reads_Oct22.sh | sh > "../data/scratch/Oct22/pbmm2_output.txt" 2>&1
```

This is a common pattern that lets you inspect the commands before running them.

### Output

12 aligned BAM files in `../data/scratch/Oct22/`, one per barcode. Each read inside now has **alignment coordinates** — a record of exactly how it maps to the parent, encoded as a CIGAR string.

---

## Stage 2: `process_bams_Oct22.py`

**Purpose:** Open each aligned BAM, filter reads aggressively for quality, and extract the mutations relative to the parent.

### Key constants

```python
# process_bams_Oct22.py

ref_start = 53
ref_end = 875
ref_len = ref_end - ref_start   # = 822 bp, the expected gene length

min_length_past_ends = 1   # at least 1 bp of flanking sequence on each side
min_phred_base = 93        # quality threshold (essentially perfect confidence)

INTERNAL_ALLOWED_CIGAR_OPS = set([CDIFF, CEQUAL])  # only substitutions allowed, no indels
```

### What is a CIGAR string?

When a read gets aligned to a reference, the alignment is stored as a **CIGAR string** — a compact notation describing how the read maps base by base. The name stands for "Compact Idiosyncratic Gapped Alignment Report."

Examples:
- `50S 822= 45S` → 50 bases of soft clip, 822 bases that match the reference exactly, 45 bases of soft clip
- `50S 400= 1X 421= 45S` → same, but one base in the middle differs from the reference (that's a mutation)

**Soft clipping** (the `S`) means the aligner found extra sequence on the ends that doesn't match the reference. These are the flanking vector/barcode sequences — the aligner trims them by marking them as "soft clipped" rather than deleting them.

In `pysam`, CIGAR operations are represented as numeric codes: `CSOFT_CLIP`, `CEQUAL` (match), `CDIFF` (substitution).

### What is a Phred quality score?

Each base in a read has a confidence score. **Phred scores** encode this on a logarithmic scale:
- Phred 10 = 1 in 10 chance of error (90% accurate)
- Phred 20 = 1 in 100 (99%)
- Phred 30 = 1 in 1,000 (99.9%)
- Phred 93 = ~1 in 2 billion

Sameer requires **every single base** to have Phred ≥ 93. This is an extremely strict cutoff that only CCS reads can achieve (because each molecule was read many times in a circle, driving the error rate down).

### The main loop

The script iterates through every read in each BAM file and applies three sequential filters:

```python
# process_bams_Oct22.py — main loop (simplified for clarity)

for i, line in enumerate(samfile):
    ct = line.cigartuples   # e.g., [(CSOFT_CLIP, 50), (CEQUAL, 400), (CDIFF, 1), (CEQUAL, 421), (CSOFT_CLIP, 45)]
```

**Filter 1 — Alignment structure:**
```python
    # Both ends must be soft-clipped (confirming the read spans the full gene
    # with flanking sequence on each side)
    if ((ct[0][0] != CSOFT_CLIP) and (ct[-1][0] != CSOFT_CLIP)):
        continue

    start = ct[0][1]  # length of the leading soft clip

    # Must have at least 1 base of flanking on each end
    if ((start < min_length_past_ends)
            or (ct[-1][1] < min_length_past_ends)):
        continue

    # Internal operations can ONLY be matches (CEQUAL) or substitutions (CDIFF)
    # This removes any reads with insertions or deletions
    if (any(x not in INTERNAL_ALLOWED_CIGAR_OPS for x, _ in ct[1:-1])):
        continue

    align_filter_counter += 1
```

**Filter 2 — Length:**
```python
    # The aligned region between the soft clips must be exactly 822 bp
    read_len = sum(x for _, x in ct[1:-1])
    if read_len != ref_len:
        continue

    length_filter_counter += 1
```

**Filter 3 — Quality:**
```python
    # Every base in the aligned region must have Phred >= 93
    end = start + read_len
    qual = np.array(line.get_forward_qualities()[start:end])
    if np.any(qual < min_phred_base):
        continue

    quality_filter_counter += 1
```

### Extracting mutations

For reads that pass all three filters, the script extracts the 822 bp gene sequence and compares it to the parent, both at the DNA and amino acid level:

```python
    seq = line.seq[start:end]                  # the 822 bp gene sequence
    ss_dna = shorten_seq(seq, wt_dna)          # mutation shorthand at DNA level
    dna_dist = hamming_dist(seq, wt_dna)       # number of DNA differences

    seq_translate = Bio.Seq.Seq(seq).translate()  # translate to amino acids
    ss_aa = shorten_seq(seq_translate, wt)        # mutation shorthand at AA level
    aa_dist = hamming_dist(seq_translate, wt)     # number of AA differences
```

The `shorten_seq` function creates the compact mutation notation:

```python
# process_bams_Oct22.py

def shorten_seq(s, wt):
    assert(len(s) == len(wt))   # no deletions — must be same length
    # For each position where they differ, record "WTbase POSITIONnewbase"
    ret = ",".join(f"{wi}{i+1}{si}" for i, (si, wi)
                   in enumerate(zip(s, wt)) if wi != si)
    if not len(ret):       # if no differences, it's the parent
        ret = f"*0*"       # special code for parent sequence
    return ret
```

So `G43T,T454C,T594A` means: position 43 changed from G to T, position 454 from T to C, position 594 from T to A. And `*0*` means this read is identical to the parent.

### Output

For each barcode, a TSV file like `bc2055.tsv` with columns:

| ss_dna | dna_dist | ss_aa | aa_dist | ref_dist |
|--------|----------|-------|---------|----------|
| G43T,T454C,T594A | 3 | A15S,F152L,N198K | 3 | 3 |
| A679G | 1 | K227E | 1 | 1 |
| *0* | 0 | *0* | 0 | 0 |

Plus a summary file `Samples.csv` tracking how many reads survived each filter:

```python
# process_bams_Oct22.py

print("barcode,sample,num_placed,num_seqs,num_kept,"
      "align_filter,length_filter,quality_filter",
      file=fh_out)
```

---

## Stage 3: `nb_preprocess_bam_Oct22.ipynb`

**Purpose:** Merge all 12 per-barcode TSVs, aggregate duplicate reads, filter further, and produce the final dataset.

### Loading and aggregating (Cells 3, 7–9)

First, it reads the summary file:

```python
# Cell 3
samples = pd.read_csv(data_dir / "Samples.csv")
samples[["parent", "category"]] = samples["sample"].str.split("_", expand=True)
```

Then it reads all 12 per-barcode TSVs and counts how many reads share the same DNA sequence within each barcode:

```python
# Cell 7
value_counts_d = {}
for filename in data_dir.glob("*.tsv"):
    x = pd.read_csv(filename, sep="\t")
    y = x.groupby("ss_dna").agg(
     dna_count = ('ss_dna', len),     # how many reads have this exact sequence
     dna_dist = ('dna_dist', lambda x: x.head(1)),
     aa_dist = ('aa_dist', lambda x: x.head(1)),
     ss_aa = ('ss_aa', lambda x: x.head(1)),
    ).sort_values(by='dna_count', ascending=False).reset_index()
    y['early_stop'] = y.ss_aa.str.contains("\*") & (y.ss_aa != "*0*")
    value_counts_d[filename.stem] = y
```

This is the critical step. If 20,000 reads all had the exact same sequence `G43T,T454C,T594A`, they collapse into one row with `dna_count = 20356`. The `dna_count` column is your confidence metric — high count means the variant is real, low count means it might be noise.

The `early_stop` flag checks whether the amino acid sequence contains a `*` (stop codon) that isn't the `*0*` parent marker. A stop codon mid-sequence means the protein gets cut short — biologically it shouldn't be functional.

Everything gets merged and labeled with parent and bin:

```python
# Cell 9
all_seqs = pd.concat([add_parent_and_category(barcode, vcs) for barcode, vcs
                       in value_counts_d.items()])
all_seqs["parent"] = pd.Categorical(all_seqs.parent,
                     categories=["1VH", "2L", "3VRL"], ordered=True)
all_seqs["category"] = pd.Categorical(all_seqs.category,
                       categories=["H", "P", "L", "N"], ordered=True)
```

### Finding the parent contamination problem (Cell 13)

This is where the problem from your notes (step 9) becomes visible:

```python
# Cell 13
parent_locations_df = all_seqs[all_seqs["ss_dna"] == "*0*"]
```

This finds every instance where the parent sequence (`*0*`) appears across all bins. The parent *should* mostly appear in the P (Parent) bin. Finding it as the #1 most abundant sequence in 1VH-High (with 11,321 reads) or in 2L-Low is the contamination issue — the parent got into bins where it shouldn't be.

### Filter 1 — Top N per bin (Cell 18)

Christian placed a known number of variants in each bin. Keep only that many:

```python
# Cell 18
placed_cutoff_seqs = pd.concat(
    g.head(get_num_samples_placed(parent, category))
    for (parent, category), g in all_seqs.groupby(["parent", "category"])
)
```

For 1VH-High, Christian placed 4 variants, so only the top 4 sequences (by read count) survive. For 2L-Low, 532 placed → top 532 kept. This immediately cuts the long tail of low-confidence sequences.

### Filter 2 — Read count cutoff (Cells 15, 19)

Even within the top N, there's a visible dropoff in read counts. The notebook sets cutoffs per bin:

```python
# Cell 15
samples_placed_df[["dna_cutoff"]] = 100    # default: need at least 100 reads

# Override for bins with suspicious patterns
samples_placed_df.loc[
    (samples_placed_df.parent == "1VH") &
    (samples_placed_df.category == "H"), "dna_cutoff"] = 10000

samples_placed_df.loc[
    (samples_placed_df.parent == "1VH") &
    (samples_placed_df.category == "P"), "dna_cutoff"] = 10000

samples_placed_df.loc[
    (samples_placed_df.parent == "2L") &
    (samples_placed_df.category == "H"), "dna_cutoff"] = 1000
```

Note how 1VH-H and 1VH-P get a much higher cutoff (10,000 reads minimum). This is because those bins had very few variants placed (4 and 2), but the sequencing returned tons of low-count junk. The high cutoff aggressively removes noise.

```python
# Cell 19
dna_cutoff_seqs = pd.concat(
    g[g.dna_count >= get_dna_cutoff(parent, category)]
    for (parent, category), g in placed_cutoff_seqs.groupby(["parent", "category"])
)
```

### Filter 3 — Mutation distance (Cells 20–21)

Remove anything with 12+ DNA mutations from the parent:

```python
# Cell 20-21
dna_distance_cutoff = 12
dist_cutoff_seqs = dna_cutoff_seqs[dna_cutoff_seqs.dna_dist < dna_distance_cutoff]
```

Error-prone PCR typically introduces only a few mutations per variant. Something with 12+ is almost certainly a chimera (two partial molecules that got fused during PCR), a severely misread sequence, or contamination from an unrelated sample.

### Filter 4 — Early stop codons (Cell 27)

Remove sequences that would produce a truncated protein:

```python
# Cell 27
no_truncated_seqs = dist_cutoff_seqs[~dist_cutoff_seqs.early_stop]
```

This is where the 3VRL problem shows up — some sequences with early stops were landing in the High activity bin, which should be impossible since a truncated protein shouldn't be functional. Those get removed here.

### Writing the final output (Cell 28)

```python
# Cell 28
no_truncated_seqs.to_csv("../data/sequences_Oct22.tsv", sep="\t", index=False)
```

---

---

## Stage 4: `nb_screening_data.ipynb`

**Purpose:** This notebook lives on the *wet-lab side* of the project. While stages 1–3 answered "what DNA sequence is in each bin?", this notebook answers "what did Christian actually *measure* for each well on each plate?" It loads two Excel files — one with the raw activity measurements and one with the bin assignments — joins them together, and does quality-control analysis. This is the bridge between the sequencing pipeline and the actual experimental results.

### The two Excel files

#### `2022-11-16 Screening data corrected.xlsx` — The activity measurements

This file has four sheets, one per library (1-VH, 2-L, 3-VRL, 2-D). Each row is one well on one plate — a single physical location where one variant was expressed and assayed.

The columns:

| Column | Meaning |
|--------|---------|
| `Sample` | Unique well ID, e.g. `1-VH_01A-01_001` (library, plate-row-col, sequential #) |
| `Plate` | Which 96-well plate this sample was on (integer) |
| `Row` | Row on the plate (A through H) |
| `Column` | Column on the plate (1 through 12) |
| `Variant?` | `1` = a library variant, `0` = a parent control, `-1` = blank/empty well |
| `Area(S)` | Raw peak area for substrate (from mass spec or chromatography) |
| `Area(OH1)` | Raw peak area for hydroxylation product 1 |
| `Area(OH2)` | Raw peak area for hydroxylation product 2 |
| `Area(N3)` | Raw peak area for azidation product |
| `%OH1`, `%OH2` | Hydroxylation percentages (product / total) |
| `%N3` | **Azidation percentage** — the main activity metric |
| `N3/OH` | **Chemoselectivity** — ratio of azidation to hydroxylation |
| `%N3(plate parent)` | Average parent %N3 on this plate (for normalization) |
| `Std Dev` | Standard deviation of parent controls on this plate |

The actual measured activity is in `%N3` (what fraction of substrate was converted to the azidation product) and `N3/OH` (how selective the enzyme is for azidation over hydroxylation). These are what "activity" means throughout the project.

**What is a lysate assay?** The variants aren't purified proteins — they're expressed in bacterial cells, the cells are broken open ("lysed"), and the raw cell extract (lysate) is used directly in the assay. This is fast (you can screen ~900 variants) but noisy compared to purified protein. That's why your notes say purified protein is "the most trustworthy measurement" — only ~10 hand-picked variants got that treatment.

**Parent controls:** On every plate, some wells contain the known parent enzyme instead of a variant. These are marked `Variant? = 0` (there are 36 per library sheet). They serve as internal controls — if the parent's activity drifts between plates, you know there's a batch effect. The `%N3(plate parent)` column gives the plate-average parent activity for normalization.

There's a "corrected" version because some samples had wrong plate numbers in the original:

```python
# The correction fixed plate assignments for 12 rows (e.g., plate 4 → plate 3)
# and changed the Variant? column from integers to strings (adding 'X' as a new category)
```

#### `2022-11-21 Bins.xlsx` — The bin assignments

This file also has four sheets (1-VH, 2-L, 3-VRL, 2-D). Each sheet has four columns: `High`, `Parent`, `Low`, `No Activity`. The values are plate-row-column strings:

```
     High    Parent    Low       No Activity
0    1G-12   5A-09     1A-05     1A-01
1    6G-06   6A-11     1A-06     1A-02
2    8B-03   NaN       1A-10     1A-03
3    8H-01   NaN       1A-11     1A-04
```

This is Christian's classification. After running the lysate assay, he looked at the activity levels and sorted each variant into one of the four bins. `1G-12` means plate 1, row G, column 12. The NaN values mean that column ran out of entries — for 1-VH, only 4 variants were "High" and only 2 were "Parent-like."

These counts match exactly what's hardcoded in `process_bams_Oct22.py`:

```
1-VH:  High=4,  Parent=2,   Low=349, No Activity=454  → 809 variants total
2-L:   High=29, Parent=52,  Low=532, No Activity=287  → 900 variants total
3-VRL: High=81, Parent=222, Low=266, No Activity=330  → 899 variants total
```

### The notebook code

#### Loading screening data (Cell 4)

```python
# Cell 4

# Load the Excel — one sheet per library, columns A through P
screen_d = {p: pd.read_excel(xlsx_dir / "2022-11-16 Screening data corrected.xlsx",
                             usecols="A:P", sheet_name=i) for i, p in enumerate(parents)}

# parents = ["1-VH", "2-L", "3-VRL", "2-D"]

# Remove rows with no sample (empty wells)
screen_d = {k: v[~v["Sample"].isna()] for k, v in screen_d.items()}

# Create a standardized plate-row-col identifier for joining
# e.g., "01-A-01" for plate 1, row A, column 1
for k, v in screen_d.items():
    v["plate_row_col"] = (v["Plate"].astype(int).astype(str).str.zfill(2) + "-" +
                          v["Row"] + "-" +
                          v["Column"].astype(int).astype(str).str.zfill(2))
```

#### Loading bin assignments (Cell 6)

The bin file has a wide format (one column per bin). The notebook melts it into a long format:

```python
# Cell 6

bins_d = {p: pd.read_excel(xlsx_dir / "2022-11-21 Bins.xlsx", usecols="A:D",
                           sheet_name=i) for i, p in enumerate(parents)}

# Melt from wide (High, Parent, Low, No Activity as columns)
# to long (one row per variant with "category" and "plate_row_col")
bins_d = {p: pd.melt(v, var_name="category", value_name="plate_row_col")
              for p, v in bins_d.items()}

# Remove NaN entries (the ragged ends of each column)
bins_d = {p: v[~v["plate_row_col"].isna()] for p, v in bins_d.items()}
```

The plate-row-col strings in the bins file use a slightly different format (`1G-12` instead of `01-G-12`), so the notebook standardizes them:

```python
# Cell 7

def standardize_plate_row_col(s):
    """Take the string for plate_row_col and make it look like 05-A-11"""
    assert(s.count("-") == 1)  # there should be exactly one - in it already
    sp = s.split("-")
    row = sp[0][-1]            # last character of first part is the row letter
    plate = int(sp[0][:-1])    # everything before that is the plate number
    col = int(sp[1])           # second part is the column
    return f"{plate:02d}-{row}-{col:02d}"

# Apply to all bins
for k, v in bins_d.items():
    v["plate_row_col"] = v["plate_row_col"].map(standardize_plate_row_col)
```

Now `1G-12` becomes `01-G-12`, matching the format from the screening data.

#### The join (Cell 9)

This is the critical step — connecting activity measurements to bin assignments:

```python
# Cell 9

# Outer join: every screening row gets a bin label, every bin entry gets its measurements
join_d = {}
for p in parents:
    join_d[p] = pd.merge(screen_d[p], bins_d[p], on=["plate_row_col"], how="outer")

# Sanity check: every bin entry should have found a matching screening row
for p in parents:
    assert(len(join_d[p][join_d[p]["Sample"].isna()][["plate_row_col", "category"]]) == 0)
```

After this join, each row has both its measured activity (`%N3`, `N3/OH`) AND its bin label (`High`, `Parent`, `Low`, `No Activity`). Some variants don't have a bin assignment (they were screened but not classified) — these have `category = NaN`.

#### Checking for unmatched variants (Cell 10)

```python
# Cell 10

# Which variants (Variant?=1) have no bin assignment?
for p in parents:
    print(p)
    print(join_d[p][join_d[p]["category"].isna() & (join_d[p]["Variant?"].astype(str) == "1")])
```

This prints any variants that were screened but never placed into a bin — these fall through the cracks and won't appear in the sequencing data.

#### Counting what's in each bin (Cell 11)

```python
# Cell 11

# Cross-tabulate: how many variants per library per bin?
counts = []
for p in parents:
    v = join_d[p][~join_d[p]["category"].isna()]
    vc = v["category"].value_counts()
    vc = pd.DataFrame(vc).reset_index()
    vc.columns = ["category", "count"]
    vc["parent"] = p
    counts.append(vc)
pd.concat(counts).pivot_table(index=["parent"], columns=["category"],
                        aggfunc=sum, margins=True, margins_name="Sum")
```

This produces the table confirming the bin counts match between the two Excel files.

#### Quality control: parent stability across plates (Cells 14–15)

This is where the notebook checks for **batch effects** — systematic differences between plates:

```python
# Cell 14

# Filter to parent controls only (Variant? = 0)
parent_df = all_d[all_d["Variant?"].astype(str) == "0"]

# Box plot: parent chemoselectivity (N3/OH) by plate, per library
g = sns.catplot(data=parent_df,
            x="Plate", y="N3/OH", col="parent", col_wrap=2, kind="box",
            sharex=True, sharey=False)
g.map(sns.swarmplot, "Plate", "N3/OH", color="k", order=sorted(all_d.Plate.unique()))
```

If the parent's activity is consistent across plates, the boxes will be tight and at similar heights. If they jump around, that means the experimental conditions varied between plates, which makes comparing variants across plates unreliable.

```python
# Cell 15

# Same thing but for azidation activity (%N3)
g = sns.catplot(data=parent_df,
            x="Plate", y="%N3", col="parent", col_wrap=2, kind="box",
            sharex=True, sharey=False)
g.map(sns.swarmplot, "Plate", "%N3", color="k", order=sorted(all_d.Plate.unique()))
```

#### Variants overlaid on parent controls (Cell 16)

```python
# Cell 16

# Strip plot of all variants colored by bin, with parent box plots underneath
g = sns.catplot(data=all_d[all_d["Variant?"].astype(str) == "1"],
            x="Plate", y="%N3", col="parent", col_wrap=2, kind="strip",
            sharex=True, sharey=False, hue="category")

for parent, ax in g.axes_dict.items():
    data = parent_df[parent_df.parent.astype(str) == parent]
    sns.boxplot(data=data, y="%N3", x="Plate", ax=ax, boxprops=dict(alpha=.6))
```

This overlays the variant measurements (colored dots: High=one color, Low=another, etc.) on top of the parent box plots. It lets you visually check whether "High" variants are actually above the parent and "Low" variants are actually below. If the bin assignments are good, you should see clear color separation.

### What this notebook tells you

This notebook doesn't produce an output file that feeds into the pipeline. Instead, it's **the evidence that the bin assignments are meaningful**. The whole ML training setup depends on trusting that "High" really means high activity and "Low" really means low. These plots let you verify that:

1. Parent controls are stable enough across plates that plate-to-plate variation isn't dominating the signal.
2. The bin assignments actually correspond to measured activity differences.
3. The `num_placed` counts from the bins file match what was actually screened.

It also connects the sequencing pipeline's sample counts (from `Samples.csv`) to the screening data's bin counts, confirming nothing was lost in translation.

### How this relates to the 1,326 number

Your notes mention a final dataset of 1,326 used to train the MLP. The `sequences_Oct22.tsv` has 1,490 rows. The gap is likely from a later step (not in these files) where Sameer further filters by removing parent sequences from bins they shouldn't be in, removing remaining duplicates across bins (the same variant appearing in multiple bins), or joining with the screening data and dropping variants that can't be matched. The screening data notebook is the foundation for that join — it establishes which plate-row-col belongs to which bin with which activity measurement.

---

## The final file: `sequences_Oct22.tsv`

**1,490 rows** (plus header). Each row is one unique variant:

| Column | Meaning | Example |
|--------|---------|---------|
| `ss_dna` | DNA mutations vs parent | `G43T,T454C,T594A` |
| `dna_count` | Number of reads supporting this sequence | `20356` |
| `dna_dist` | Number of DNA bases different from parent | `3` |
| `aa_dist` | Number of amino acids different from parent | `3` |
| `ss_aa` | Amino acid mutations vs parent | `A15S,F152L,N198K` |
| `early_stop` | Has premature stop codon (always False here) | `False` |
| `seq_rank` | Rank by read count within its bin | `1` |
| `parent` | Which library (1VH, 2L, or 3VRL) | `1VH` |
| `category` | Activity bin (H, P, L, N) | `H` |

### Composition

| Library | H | P | L | N | Total |
|---------|---|---|---|---|-------|
| 1VH | 4 | 1 | 291 | 161 | 457 |
| 2L | 23 | 37 | 337 | 130 | 527 |
| 3VRL | 50 | 139 | 146 | 171 | 506 |
| **Total** | **77** | **177** | **774** | **462** | **1,490** |

Note: the parent sequence (`*0*`) appears in this file. It shows up 7 times across different bins (e.g., it's seq_rank 4 in 1VH-H with 11,321 reads, and seq_rank 1 in 2L-H with 6,378 reads). This is the contamination issue — the parent leaked into bins where it theoretically shouldn't be.

## Files covered in this walkthrough

| File | Stage |
|------|-------|
| `align_pacbio_reads_Oct22.sh` | Stage 1 — Alignment |
| `process_bams_Oct22.py` | Stage 2 — BAM processing and filtering |
| `nb_preprocess_bam_Oct22.ipynb` | Stage 3 — Aggregation and final filtering |
| `nb_screening_data.ipynb` | Stage 4 — Wet-lab measurements and QC |
| `2022-11-16 Screening data corrected.xlsx` | Input to Stage 4 — activity measurements |
| `2022-11-21 Bins.xlsx` | Input to Stage 4 — bin assignments |
| `2022-11-16 Screening data.xlsx` | Uncorrected version (12 plate numbers fixed in corrected version) |