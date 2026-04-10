"""
DCA Score Box Plots by Category and Parent Library.

Produces a faceted box plot showing DCA score distributions across
activity categories (H, P, L, N) for each parent library.

Usage:
    python plot_dca_boxplots.py output/ordinal_Oct22_sequences_with_dca_score.csv
"""

import sys
import pathlib
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr


def setup_logging(output_dir):
    """Configure logging to both terminal and file."""
    logger = logging.getLogger("plot_dca")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s — %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = output_dir / "plot_dca_boxplots.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def create_output_dir(base_dir="output/model_evals_analysis"):
    """Create a timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = pathlib.Path(base_dir) / f"dca_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main(data_csv):
    output_dir = create_output_dir()
    logger = setup_logging(output_dir)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data file: {data_csv}")

    # load data
    df = pd.read_csv(data_csv)
    logger.info(f"Loaded {len(df)} sequences")
    logger.info(f"Parents: {df.parent.unique().tolist()}")
    logger.info(f"Categories: {df.category.unique().tolist()}")

    # log counts per group
    for parent in df.parent.unique():
        counts = df[df.parent == parent].category.value_counts()
        logger.info(f"  {parent}: {counts.to_dict()}")

    # ordering and colors
    category_order = ["N", "L", "P", "H"]
    category_labels = {"N": "Dead", "L": "Low", "P": "Parent", "H": "High"}
    parents = ["1VH", "2L", "3VRL"]
    category_colors = {
        "N": "#C44E52",
        "L": "#55A868",
        "P": "#DD8452",
        "H": "#4C72B0",
    }

    # font setup
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
    })

    fig, axes = plt.subplots(len(parents), 1, figsize=(8, 12),
                              sharex=True)

    category_to_num = {"N": 0, "L": 1, "P": 2, "H": 3}

    for ax, parent in zip(axes, parents):
        parent_df = df[df.parent == parent]

        # compute spearman correlation between DCA score and ordinal category
        rho, pval = spearmanr(parent_df["dca_score"],
                              parent_df["category"].map(category_to_num))

        box_data = []
        positions = []
        colors = []
        for i, cat in enumerate(category_order):
            cat_data = parent_df[parent_df.category == cat]["dca_score"]
            box_data.append(cat_data.values)
            positions.append(i)
            colors.append(category_colors[cat])

        bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='d', markersize=4,
                                        markerfacecolor='#555555',
                                        markeredgecolor='#555555'),
                        medianprops=dict(color='#2C3E50', linewidth=1.5),
                        whiskerprops=dict(color='#2C3E50', linewidth=1.0),
                        capprops=dict(color='#2C3E50', linewidth=1.0))

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('#2C3E50')
            patch.set_linewidth(1.0)
            patch.set_alpha(0.85)

        ax.set_title(f"{parent}  (Spearman r = {rho:.3f})",
                     fontsize=14, pad=10)
        ax.set_ylabel("DCA score (lower is fitter)", fontsize=12)
        ax.set_facecolor("#EAEAF2")
        ax.grid(axis='y', color='white', linewidth=1.0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis='y', labelsize=11)

    # x-axis labels on bottom plot only
    axes[-1].set_xticks(range(len(category_order)))
    axes[-1].set_xticklabels([category_labels[c] for c in category_order],
                              fontsize=13)
    axes[-1].set_xlabel("Activity Category", fontsize=13)

    plt.tight_layout()

    figure_path = output_dir / "dca_score_boxplots.png"
    plt.savefig(figure_path, dpi=150, bbox_inches="tight",
                facecolor="white", pad_inches=0.3)
    logger.info(f"Figure saved to {figure_path}")
    plt.close()

    # save summary statistics
    stats_rows = []
    for parent in parents:
        parent_df = df[df.parent == parent]
        for cat in category_order:
            cat_data = parent_df[parent_df.category == cat]["dca_score"]
            stats_rows.append({
                "parent": parent,
                "category": cat,
                "count": len(cat_data),
                "mean": cat_data.mean(),
                "median": cat_data.median(),
                "std": cat_data.std(),
                "min": cat_data.min(),
                "max": cat_data.max(),
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = output_dir / "dca_score_summary.csv"
    stats_df.to_csv(stats_csv, index=False)
    logger.info(f"Summary statistics saved to {stats_csv}")

    logger.info("Done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_dca_boxplots.py <data_csv>")
        sys.exit(1)
    main(sys.argv[1])