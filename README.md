# SadX Latent Sequence Function

Open-source implementation for generating and prioritizing SadX enzyme variants using typically discarded data from directed evolution libraries. This repository provides the code and data needed to regenerate the results described in the SadX paper.

> **Note:** This repository only reproduces the results from the SadX latent sequence function paper and is not optimized for any other protein. 

## Overview

The project uses screening data from three rounds of directed evolution on SadX (libraries 1-VH, 2-L, 3-VRL) — data that is typically discarded after each round. Variants were sequenced on PacBio CCS, screened in lysate for azidation activity, binned into four ordinal activity categories (High, Parent-like, Low, Dead), and processed into a training set of 1,326 variants. An MLP trained with CORAL loss on this data was then used to recommend new single-mutant variants on top of the 3-VRL parent. The repository is organized around four reproducible components:

| # | Component | What it does |
|---|-----------|-------------|
| 1 | [Sequence Preprocessing](#1-sequence-preprocessing) | PacBio CCS → [1,326 filtered variants](output/ordinal_Oct22_sequences_with_dca_score.csv) ([preprocessing README](scripts/process_sequence_generation/README.md)) |
| 2 | [DCA Scoring](#2-dca-scoring) | Generates the MSA and computes bmDCA co-evolutionary density scores for each 1,326 variants ([DCA README](scripts/bmDCA_sadA/README.md)) |
| 3 | [Model Training](#3-model-training) | Trains CORAL MLP for ordinal activity prediction using the [final model configuration](output/ordinal_Oct22_models/saved_models/6eeae50e.yml) ([model README](output/ordinal_Oct22_models/README.md)) |
| 4 | [Retrospective Feature Importance](#4-retrospective-feature-importance) | Ablation over DCA and multi-library features using Codex ([ablation README](cv_balanced_accuracy/README.md)) |

---

## 1. Sequence Preprocessing

The [`scripts/process_sequence_figures_generation/README.md`](scripts/process_sequence_figures_generation/README.md) readme describes how to go from the raw reads to the full sequence-to-activity dataset after going through a series of quality filters.

Raw sequences can be downloaded from the SRA (PacBio circular consensus sequencing reads have been deposited in the NCBI Sequence Read Archive under accession [PRJNA1505729](https://www.ncbi.nlm.nih.gov/sra/PRJNA1505729)).

Intermediate versions of the data — demultiplexed, quality-filtered reads from each processing step — are saved in the repository. This allows one to run the downstream analysis without having to reprocess from raw reads.

The final dataset is here: `output/ordinal_Oct22_sequences_with_dca_score.csv`

---

## 2. DCA Scoring

Generating the multiple sequence alignment (MSA) and bmDCA co-evolutionary density scores used as an additional input feature for the CORAL model. See [`scripts/bmDCA_sadA/README.md`](scripts/bmDCA_sadA/README.md) for details on MSA construction and DCA score computation.

> **Note:** The DCA scores for the training set are provided pre-computed in `output/ordinal_Oct22_sequences_with_dca_score.csv`. The model can be trained using these pre-computed scores without re-running bmDCA.

---

## 3. Model Training

How to run the final model including the model configuration, predictions, and outputs are documented in [`output/ordinal_Oct22_models/README.md`](output/ordinal_Oct22_models/README.md).

The final model is a two-hidden-layer MLP (100 → 50 units, dropout 0.2) trained with CORAL loss for ordinal classification into four activity categories. Architecture decisions were motivated by literature review:

- **CORAL loss** for ordinal regression (respects the ordered relationship between N < L < P < H)
- **Multi-library bias terms** (library-specific intercepts in the output layer, accounting for different activity thresholds across the three parent libraries)
- **bmDCA feature** (co-evolutionary density score appended to the one-hot encoding)

The model was trained on the full dataset for 400 epochs with learning rate 3e-4 and weight decay 1e-6.

---

## 4. Retrospective Feature Importance

A retrospective ablation study over model features (DCA, multi-library bias) is documented in [`cv_balanced_accuracy/README.md`](cv_balanced_accuracy/README.md). This analysis uses cross-validated balanced accuracy to explore the contribution of each feature to the final model's performance.