#!/usr/bin/env python3
"""
Generate a supplementary table figure showing model-predicted single mutations,
their observations in the DMS training data, and other substitutions observed
at each position.

Supports two modes:
  1. Top-N by P(High) (default) — set CUSTOM_MUTATIONS = None
  2. Custom mutation list        — set CUSTOM_MUTATIONS to a list of dicts

Each custom mutation can optionally restrict which parent libraries are searched.

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
from matplotlib.colors import to_rgba

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
DMS_CSV = "output/ordinal_Oct22_sequences_with_dca_score.csv"
HBIN_TSV = "output/ordinal_Oct22_models/saved_models/6eeae50e.Hbin_probs.tsv"
OUTPUT_DIR = "output/singles_train_top_N"
TOP_N = 10

# ── Custom mutations ─────────────────────────────────────
# Set to None to use top-N mode (ranked by P(High) from Hbin predictions).
#
# Otherwise, provide a list of dicts, each with:
#   "mutation" : str   — e.g. "I38V" (wt_aa + position + mut_aa)
#   "origin"   : str   — which parent this mutation defines / created.
#                         e.g. "2L" means this mutation is part of the
#                         starting sequence for round 2 (2L).  Use "—" if N/A.
#   "parents"  : list  — (optional) parent libraries to search for
#                         DMS observations.  Omit or None → search all.

CUSTOM_MUTATIONS  = None
# CUSTOM_MUTATIONS = [
#     {"mutation": "F152L", "origin": "2L",   "parents": ["1VH"]},
#     {"mutation": "I38V",  "origin": "3VRL",   "parents": ["1VH", "2L"]},
#     {"mutation": "Q233R", "origin": "3VRL",   "parents": ["1VH", "2L"]},
#     {"mutation": "F261L", "origin": "3VRL",   "parents": ["1VH", "2L"]},
#     {"mutation": "V38I",  "origin": "4IC",  "parents": ["3VRL"]},
#     {"mutation": "R48C",  "origin": "4IC",  "parents": ["1VH", "2L", "3VRL"]},
# ]

# CUSTOM_MUTATIONS= None
ALL_PARENTS = ["1VH", "2L", "3VRL"]

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

# Row background colours by origin (low alpha)
ORIGIN_COLORS = {
    "2L":   to_rgba("#4488cc", alpha=0.10),   # blue
    "3VRL": to_rgba("#44aa66", alpha=0.10),   # green
    "4IC":  to_rgba("#dd8833", alpha=0.10),   # orange
}
ORIGIN_DEFAULT = to_rgba("#999999", alpha=0.06)  # fallback for "—" or unknown

# ──────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def parse_mutation(mut_str):
    """Parse a mutation string like 'I38V' into (wt_aa, pos, mut_aa)."""
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos = int(mut_str[1:-1])
    return wt_aa, pos, mut_aa


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
def find_exact_observations(df, parent_refs, pos, mut_aa, parents=None):
    """Find every DMS variant carrying the exact substitution at `pos`.

    Parameters
    ----------
    parents : list of str or None
        Which parent libraries to search.  None means all three.

    Returns list of (description_string, is_single) tuples.
    """
    if parents is None:
        parents = ALL_PARENTS

    idx = pos - 2
    results = []

    for parent_name in parents:
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
                desc = f"{parent_name}: single → {cat}"
            else:
                others = [f"{w}{p}{m}" for p, w, m in diffs if p != pos]
                desc = f"{parent_name}: {n_muts}-mutant → {cat} (+{', '.join(others)})"

            results.append((desc, is_single))

    return results


def find_positional_observations(df, parent_refs, wt_trim, pos, exclude_mut,
                                 parents=None):
    """Find all OTHER substitutions at `pos` (not `exclude_mut`).

    Parameters
    ----------
    parents : list of str or None
        Which parent libraries to search.  None means all three.

    Returns list of summary strings like "Y131C: 8×, best = High".
    """
    if parents is None:
        parents = ALL_PARENTS

    idx = pos - 2
    wt_aa = wt_trim[idx]

    # Collect: {mut_identity -> [categories]}
    sub_cats = defaultdict(list)

    for parent_name in parents:
        ref = parent_refs[parent_name]
        parent_wt = ref[idx]
        sub = df[df["parent"] == parent_name]

        for _, row in sub.iterrows():
            seq = row["sequence_aa_trim"]
            if seq[idx] == parent_wt:
                continue
            mut_aa = seq[idx]
            identity = f"{parent_wt}{pos}{mut_aa}"
            if mut_aa == exclude_mut:
                continue
            sub_cats[identity].append(row["category"])

    if not sub_cats:
        return ["— none —"]

    # Sort by count descending
    results = []
    for identity in sorted(sub_cats, key=lambda k: len(sub_cats[k]), reverse=True):
        cats = sub_cats[identity]
        best_cat = max(cats, key=lambda c: CAT_RANK[c])
        best_label = CAT_LABEL[best_cat]
        results.append(f"{identity}: {len(cats)}×, best = {best_label}")

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

    # ── Detect mode ──────────────────────────────────────
    is_custom = any(r.get("origin") is not None for r in table_rows)

    if is_custom:
        # Custom mode: Starting Seq | Mutation | Searched | Exact | Positional
        col_w = [1.00, 0.90, 1.10, 4.50, 3.50]
        headers = ["Defines", "Mutation", "Searched\nLibrary",
                   "Exact mutation in DMS", "Other substitutions at position"]

        # Font sizes (bigger for custom)
        HDR_FS = 13.5
        CELL_FS = 15.0
        MUT_FS = 15.0
        DETAIL_FS = 12.0
        LINE_H = 0.32
        PAD_T = 0.14
        PAD_B = 0.12
        HDR_H = 0.55
    else:
        # Top-N mode (original): Rank | Mutation | P(High) | Exact | Positional
        has_phigh = any(r["ph"] is not None for r in table_rows)
        if has_phigh:
            col_w = [0.50, 0.85, 0.70, 4.30, 3.50]
            headers = ["Rank", "Mutation", "P(High)",
                       "Exact mutation in DMS", "Other substitutions at position"]
        else:
            col_w = [0.50, 0.85, 4.50, 3.50]
            headers = ["Rank", "Mutation",
                       "Exact mutation in DMS", "Other substitutions at position"]

        HDR_FS = 8.5
        CELL_FS = 9.5
        MUT_FS = 9.5
        DETAIL_FS = 7.5
        LINE_H = 0.27
        PAD_T = 0.12
        PAD_B = 0.10
        HDR_H = 0.50

    col_x = []
    cx = 0
    for w in col_w:
        col_x.append(cx)
        cx += w
    total_w = sum(col_w)

    # Identify column indices for the multi-line cells
    exact_col_idx = headers.index("Exact mutation in DMS")
    pos_col_idx = headers.index("Other substitutions at position")

    row_heights = []
    for r in table_rows:
        n = max(len(r["exact"]), len(r["positional"]), 1)
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
                ha="center", va="center", fontsize=HDR_FS,
                fontweight="bold", color="#333")
    for i in range(1, len(col_x)):
        ax.plot([col_x[i], col_x[i]], [yt, yt - HDR_H], color="#cccccc", lw=0.5)
    ax.plot([0, total_w], [yt - HDR_H] * 2, color="#999", lw=1.0)

    # rows
    yc = yt - HDR_H
    for ri, r in enumerate(table_rows):
        rh = row_heights[ri]
        ry = yc - rh

        # ── Row background ───────────────────────────────
        if is_custom:
            origin = r.get("origin", "—")
            bg = ORIGIN_COLORS.get(origin, ORIGIN_DEFAULT)
            # Color band covers only the first 3 columns (Starting Seq → Searched)
            band_w = col_x[exact_col_idx]
            ax.add_patch(mpatches.Rectangle((0, ry), band_w, rh,
                         fc=bg, ec="none"))
        else:
            if ri % 2 == 1:
                ax.add_patch(mpatches.Rectangle((0, ry), total_w, rh,
                             fc="#f8f8f8", ec="none"))

        ax.plot([0, total_w], [ry, ry], color="#e0e0e0", lw=0.5)

        for ci in [exact_col_idx, pos_col_idx]:
            ax.plot([col_x[ci], col_x[ci]], [yc, ry], color="#e0e0e0", lw=0.5)

        my = yc - rh / 2

        if is_custom:
            # ── Custom mode columns ──────────────────────
            # Starting Seq
            seq_col = headers.index("Defines")
            ax.text(col_x[seq_col] + col_w[seq_col] / 2, my,
                    r.get("origin", "—"),
                    ha="center", va="center", fontsize=CELL_FS,
                   color="#333")

            # Mutation
            mut_col = headers.index("Mutation")
            ax.text(col_x[mut_col] + col_w[mut_col] / 2, my, r["mut"],
                    ha="center", va="center", fontsize=MUT_FS,
                    fontweight="bold", fontfamily="monospace", color="#222")

            # Searched
            srch_col = headers.index("Searched\nLibrary")
            parents_used = r.get("parents") or ALL_PARENTS
            ax.text(col_x[srch_col] + col_w[srch_col] / 2, my,
                    ",\n".join(parents_used),
                    ha="center", va="center", fontsize=CELL_FS,
                    fontfamily="monospace", color="#555")
        else:
            # ── Top-N mode columns (unchanged) ───────────
            # Rank
            ax.text(col_x[0] + col_w[0] / 2, my, str(r["rank"]),
                    ha="center", va="center", fontsize=CELL_FS, color="#333")
            # Mutation
            ax.text(col_x[1] + col_w[1] / 2, my, r["mut"],
                    ha="center", va="center", fontsize=MUT_FS,
                    fontweight="bold", fontfamily="monospace", color="#222")
            # P(High)
            if has_phigh:
                ph_col = headers.index("P(High)")
                ph_val = r["ph"]
                ph_text = f"{ph_val:.3f}" if ph_val is not None else "—"
                ax.text(col_x[ph_col] + col_w[ph_col] / 2, my, ph_text,
                        ha="center", va="center", fontsize=CELL_FS,
                        fontfamily="monospace", color="#333")

        # ── Exact mutation column (shared) ───────────────
        for li, (line, is_single) in enumerate(r["exact"]):
            ly = yc - PAD_T - LINE_H / 2 - li * LINE_H
            if is_single:
                ax.add_patch(mpatches.FancyBboxPatch(
                    (col_x[exact_col_idx] + 0.03, ly - LINE_H * 0.42),
                    col_w[exact_col_idx] - 0.06, LINE_H * 0.84,
                    boxstyle="round,pad=0.02",
                    fc=SINGLE_BG, ec="#a5d6a7", lw=0.6))
                ax.plot(col_x[exact_col_idx] + 0.14, ly, marker="D",
                        markersize=SINGLE_MARKER_SIZE, color="#2d7d3a",
                        markeredgecolor="#1b5e20", markeredgewidth=0.5,
                        zorder=5, clip_on=False)
                ax.text(col_x[exact_col_idx] + 0.24, ly, line,
                        ha="left", va="center", fontsize=DETAIL_FS,
                        fontweight="bold", color=get_color(line))
            else:
                ax.text(col_x[exact_col_idx] + 0.08, ly, line,
                        ha="left", va="center",
                        fontsize=DETAIL_FS, color=get_color(line))

        # ── Positional column (shared) ───────────────────
        for li, line in enumerate(r["positional"]):
            ly = yc - PAD_T - LINE_H / 2 - li * LINE_H
            ax.text(col_x[pos_col_idx] + 0.08, ly, line,
                    ha="left", va="center",
                    fontsize=DETAIL_FS, color=get_color(line))

        yc = ry

    # outer border
    ax.add_patch(mpatches.Rectangle((0, yc), total_w, yt - yc,
                 fc="none", ec="#bbbbbb", lw=0.8))

    plt.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.005)
    fig.savefig(save_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"PNG → {save_png}")
    logger.info(f"PDF → {save_pdf}")


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

    # ── Build the mutation list ──────────────────────────
    if CUSTOM_MUTATIONS is not None:
        # Custom mode: user-specified mutations
        logger.info("Custom mutation mode: %d mutations", len(CUSTOM_MUTATIONS))
        mutation_specs = []
        for entry in CUSTOM_MUTATIONS:
            wt_aa, pos, mut_aa = parse_mutation(entry["mutation"])
            parents = entry.get("parents", None)  # None → all parents

            # Look up P(High) if available in the Hbin file
            feat_key = f"{wt_aa}{pos}{mut_aa}"
            match = hbin[hbin["feature"] == feat_key]
            phigh = float(match["Hbin_probs"].iloc[0]) if len(match) > 0 else None

            mutation_specs.append({
                "wt_aa": wt_aa,
                "pos": pos,
                "mut_aa": mut_aa,
                "phigh": phigh,
                "parents": parents,
                "origin": entry.get("origin", "—"),
            })
    else:
        # Top-N mode (original behaviour)
        logger.info("Top-%d mode (ranked by P(High))", TOP_N)
        top = hbin.nlargest(TOP_N, "Hbin_probs")
        mutation_specs = []
        for _, row in top.iterrows():
            feat = row["feature"]
            wt_aa, pos, mut_aa = parse_mutation(feat)
            mutation_specs.append({
                "wt_aa": wt_aa,
                "pos": pos,
                "mut_aa": mut_aa,
                "phigh": row["Hbin_probs"],
                "parents": None,  # search all
            })

    # ── Collect observations for each mutation ───────────
    table_rows = []
    for rank, spec in enumerate(mutation_specs, start=1):
        wt_aa = spec["wt_aa"]
        pos = spec["pos"]
        mut_aa = spec["mut_aa"]
        phigh = spec["phigh"]
        parents = spec["parents"]
        mut_label = f"{wt_aa}{pos}{mut_aa}"

        ph_str = f"P(High)={phigh:.4f}" if phigh is not None else "P(High)=N/A"
        lib_str = ", ".join(parents) if parents else "all"
        logger.info("  #%2d  %s  %s  [%s]", rank, mut_label, ph_str, lib_str)

        exact = find_exact_observations(df, parent_refs, pos, mut_aa,
                                        parents=parents)
        positional = find_positional_observations(
            df, parent_refs, wt_trim, pos, exclude_mut=mut_aa,
            parents=parents)

        table_rows.append({
            "rank": rank,
            "mut": mut_label,
            "ph": phigh,
            "exact": exact,
            "positional": positional,
            "parents": parents,
            "origin": spec.get("origin"),
        })

    # ── Save ─────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if CUSTOM_MUTATIONS is not None:
        base = "custom_mutations_table"
    else:
        base = "top10_mutations_table"
    png_path = os.path.join(OUTPUT_DIR, f"{base}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{base}.pdf")

    plot_table(table_rows, png_path, pdf_path)
    logger.info("Done")


if __name__ == "__main__":
    main()

# adapt this code to allow for custom mutations to be visualized instead of the top 10 if i want.
# also change it so that you can determine where you can search for each mutation . so you only have to search
# in a specific library. if i have I38V , then i can say only look for occurences of this position
# and mutation in 1VH and 2L for example.

# adapt this code to allow for custom mutations to be visualized instead of the top 10 if i want.
# also change it so that you can determine where you can search for each mutation . so you only have to search
# in a specific library. if i have I38V , then i can say only look for occurences of this position
# and mutation in 1VH and 2L for example.