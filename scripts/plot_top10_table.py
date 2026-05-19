#!/usr/bin/env python3
"""
Generate a supplementary table figure showing the top-10 model-predicted
single mutations ranked by P(High), their observations in the DMS training
data, and other substitutions observed at each position.

Usage (from project root):
    python scripts/plot_top10_table.py

Inputs:
    output/ordinal_Oct22_sequences_with_dca_score.csv
    output/ordinal_Oct22_models/saved_models/6eeae50e.Hbin_probs.tsv

Outputs:
    output/singles_train_top_N/top10_mutations_table.png  (300 dpi)
    output/singles_train_top_N/top10_mutations_table.pdf  (vector)
"""

import os
import sys
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
DMS_CSV = "output/ordinal_Oct22_sequences_with_dca_score.csv"
HBIN_TSV = "output/ordinal_Oct22_models/saved_models/6eeae50e.Hbin_probs.tsv"
OUTPUT_DIR = "output/singles_train_top_N"
TOP_N = 10

WILDTYPE_SEQ_3VRL = (
    "MQHTYPAQLMRFGTAARAEHMTIAAAIHALDADEADAVVMDIVPDGERDAWWDDEGFSSSPFTK"
    "NAHHAGVVATSVTLGQLQREQGDKLVSKAAEYFGIACRVNDGLRTTRFVRLFSDALDAKPLTIG"
    "HDYEVEFLLATRRVYEPFEAPFNLAPHCGDVSYGRDTVNWPLKHSFPRQLGGFLTIQGADNDAG"
    "MVMWDNRPESRAALDEMHAEYRETGAIAALERAAKIMLKPRPGQLTLFQSKNLHAIERCTSTRR"
    "TMGLLLIHTEDGWRMFD"
)

# Parent-specific mutations relative to 3VRL (1-indexed position -> amino acid)
PARENT_DIFFS = {
    "3VRL": {},
    "1VH":  {38: "I", 152: "F", 233: "Q", 261: "F"},
    "2L":   {38: "I", 233: "Q", 261: "F"},
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

CAT_RANK = {"H": 3, "P": 2, "L": 1, "N": 0}
CAT_LABEL = {"H": "High", "P": "Parent", "L": "Low", "N": "Dead"}
CAT_COLOR = {"High": "#2d7d3a", "Parent": "#2565a0",
             "Low": "#b07d10", "Dead": "#a03030"}

SINGLE_BG = "#e8f5e9"
SINGLE_MARKER_SIZE = 5  # drawn as a matplotlib marker, not a font glyph

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s \u2014 %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Build parent reference sequences
# ──────────────────────────────────────────────────────────
def build_parent_refs():
    """Return dict {parent_name: list of AAs} for trimmed sequence (pos 2-273)."""
    wt_trim = list(WILDTYPE_SEQ_3VRL[1:])  # trim M1
    refs = {}
    for parent, diffs in PARENT_DIFFS.items():
        ref = list(wt_trim)
        for pos, aa in diffs.items():
            ref[pos - 2] = aa
        refs[parent] = ref
    return refs, wt_trim


# ──────────────────────────────────────────────────────────
# Analyze DMS observations for a given mutation
# ──────────────────────────────────────────────────────────
def find_exact_observations(df, parent_refs, pos, mut_aa):
    """Find every DMS variant carrying the exact substitution at `pos`.

    Returns list of (description_string, is_single) tuples.
    """
    idx = pos - 2
    results = []

    for parent_name in ["1VH", "2L", "3VRL"]:
        ref = parent_refs[parent_name]
        parent_wt = ref[idx]
        sub = df[df["parent"] == parent_name]

        for _, row in sub.iterrows():
            seq = row["sequence_aa_trim"]
            if seq[idx] != mut_aa:
                continue

            # count mutations relative to own parent
            diffs = [(i + 2, ref[i], seq[i])
                     for i in range(len(ref)) if seq[i] != ref[i]]
            n_muts = len(diffs)
            cat = CAT_LABEL[row["category"]]
            is_single = (n_muts == 1)

            if is_single:
                desc = f"{parent_name}: single \u2192 {cat}"
            else:
                others = [f"{w}{p}{m}" for p, w, m in diffs if p != pos]
                desc = f"{parent_name}: {n_muts}-mutant \u2192 {cat} (+{', '.join(others)})"

            results.append((desc, is_single))

    return results


def find_positional_observations(df, parent_refs, wt_trim, pos, exclude_mut):
    """Find all OTHER substitutions at `pos` (not `exclude_mut`).

    Returns list of summary strings like "Y131C: 8x, best = High".
    """
    idx = pos - 2
    wt_aa = wt_trim[idx]

    # Collect: {mut_identity -> [categories]}
    sub_cats = defaultdict(list)

    for parent_name in ["1VH", "2L", "3VRL"]:
        ref = parent_refs[parent_name]
        parent_wt = ref[idx]
        sub = df[df["parent"] == parent_name]

        for _, row in sub.iterrows():
            seq = row["sequence_aa_trim"]
            if seq[idx] == parent_wt:
                continue
            mut_aa = seq[idx]
            identity = f"{wt_aa}{pos}{mut_aa}"
            if mut_aa == exclude_mut:
                continue
            sub_cats[identity].append(row["category"])

    if not sub_cats:
        return ["\u2014 none \u2014"]

    # Sort by count descending
    results = []
    for identity in sorted(sub_cats, key=lambda k: len(sub_cats[k]), reverse=True):
        cats = sub_cats[identity]
        best_cat = max(cats, key=lambda c: CAT_RANK[c])
        best_label = CAT_LABEL[best_cat]
        results.append(f"{identity}: {len(cats)}\u00d7, best = {best_label}")

    return results


# ──────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────
def get_color(text):
    for cat in ["High", "Dead", "Low", "Parent"]:
        if cat in text:
            return CAT_COLOR[cat]
    return "#888888"


def plot_table(table_rows, save_png, save_pdf):
    """Render the table as a matplotlib figure."""

    col_w = [0.50, 0.85, 0.70, 4.30, 3.50]
    col_x = []
    cx = 0
    for w in col_w:
        col_x.append(cx)
        cx += w
    total_w = sum(col_w)

    headers = ["Rank", "Mutation", "P(High)",
               "Exact mutation in DMS", "Other substitutions at position"]

    LINE_H = 0.27
    PAD_T = 0.12
    PAD_B = 0.10
    HDR_H = 0.50

    row_heights = []
    for r in table_rows:
        n = max(len(r["exact"]), len(r["positional"]))
        row_heights.append(PAD_T + n * LINE_H + PAD_B)

    total_h = HDR_H + sum(row_heights)

    fig, ax = plt.subplots(figsize=(total_w * 1.05, total_h + 0.3))
    ax.set_xlim(-0.1, total_w + 0.1)
    ax.set_ylim(-0.1, total_h + 0.1)
    ax.axis("off")

    yt = total_h

    # header
    ax.add_patch(mpatches.Rectangle((0, yt - HDR_H), total_w, HDR_H,
                 fc="#ededed", ec="#bbbbbb", lw=0.8))
    for i, h in enumerate(headers):
        ax.text(col_x[i] + col_w[i] / 2, yt - HDR_H / 2, h,
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#333")
    for i in range(1, len(col_x)):
        ax.plot([col_x[i], col_x[i]], [yt, yt - HDR_H], color="#cccccc", lw=0.5)
    ax.plot([0, total_w], [yt - HDR_H] * 2, color="#999", lw=1.0)

    # rows
    yc = yt - HDR_H
    for ri, r in enumerate(table_rows):
        rh = row_heights[ri]
        ry = yc - rh

        if ri % 2 == 1:
            ax.add_patch(mpatches.Rectangle((0, ry), total_w, rh,
                         fc="#f8f8f8", ec="none"))

        ax.plot([0, total_w], [ry, ry], color="#e0e0e0", lw=0.5)

        for ci in [3, 4]:
            ax.plot([col_x[ci], col_x[ci]], [yc, ry], color="#e0e0e0", lw=0.5)

        my = yc - rh / 2

        ax.text(col_x[0] + col_w[0] / 2, my, str(r["rank"]),
                ha="center", va="center", fontsize=9.5, color="#333")
        ax.text(col_x[1] + col_w[1] / 2, my, r["mut"],
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                fontfamily="monospace", color="#222")
        ax.text(col_x[2] + col_w[2] / 2, my, f'{r["ph"]:.3f}',
                ha="center", va="center", fontsize=9.5,
                fontfamily="monospace", color="#333")

        # exact mutation column
        for li, (line, is_single) in enumerate(r["exact"]):
            ly = yc - PAD_T - LINE_H / 2 - li * LINE_H
            if is_single:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (col_x[3] + 0.03, ly - LINE_H * 0.42),
                    col_w[3] - 0.06, LINE_H * 0.84,
                    boxstyle="round,pad=0.02",
                    fc=SINGLE_BG, ec="#a5d6a7", lw=0.6))
                ax.plot(col_x[3] + 0.14, ly, marker="D",
                        markersize=SINGLE_MARKER_SIZE, color="#2d7d3a",
                        markeredgecolor="#1b5e20", markeredgewidth=0.5,
                        zorder=5, clip_on=False)
                ax.text(col_x[3] + 0.24, ly, line,
                        ha="left", va="center", fontsize=7.5,
                        fontweight="bold", color=get_color(line))
            else:
                ax.text(col_x[3] + 0.08, ly, line, ha="left", va="center",
                        fontsize=7.5, color=get_color(line))

        # positional column
        for li, line in enumerate(r["positional"]):
            ly = yc - PAD_T - LINE_H / 2 - li * LINE_H
            ax.text(col_x[4] + 0.08, ly, line, ha="left", va="center",
                    fontsize=7.5, color=get_color(line))

        yc = ry

    # outer border
    ax.add_patch(mpatches.Rectangle((0, yc), total_w, yt - yc,
                 fc="none", ec="#bbbbbb", lw=0.8))

    plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.005)
    fig.savefig(save_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"PNG \u2192 {save_png}")
    logger.info(f"PDF \u2192 {save_pdf}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    logger.info("Loading DMS data: %s", DMS_CSV)
    df = pd.read_csv(DMS_CSV)
    logger.info("  %d variants, %d parents", len(df), df["parent"].nunique())

    logger.info("Loading Hbin predictions: %s", HBIN_TSV)
    hbin = pd.read_csv(HBIN_TSV, sep="\t")
    logger.info("  %d predictions", len(hbin))

    parent_refs, wt_trim = build_parent_refs()

    # Top N mutations
    top = hbin.nlargest(TOP_N, "Hbin_probs")
    logger.info("Top %d mutations by P(High):", TOP_N)

    table_rows = []
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        feat = row["feature"]
        wt_aa = feat[0]
        pos = int(feat[1:-1])
        mut_aa = feat[-1]
        phigh = row["Hbin_probs"]
        mut_label = f"{wt_aa}{pos}{mut_aa}"

        logger.info("  #%2d  %s  P(High)=%.4f", rank, mut_label, phigh)

        exact = find_exact_observations(df, parent_refs, pos, mut_aa)
        positional = find_positional_observations(
            df, parent_refs, wt_trim, pos, exclude_mut=mut_aa)

        table_rows.append({
            "rank": rank,
            "mut": mut_label,
            "ph": phigh,
            "exact": exact,
            "positional": positional,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    png_path = os.path.join(OUTPUT_DIR, "top10_mutations_table.png")
    pdf_path = os.path.join(OUTPUT_DIR, "top10_mutations_table.pdf")

    plot_table(table_rows, png_path, pdf_path)
    logger.info("Done")


if __name__ == "__main__":
    main()


# adapt this code to allow for custom mutations to be visualized instead of the top 10 if i want.
# also change it so that you can determine where you can search for each mutation . so you only have to search
# in a specific library. if i have I38V , then i can say only look for occurences of this position
# and mutation in 1VH and 2L for example.