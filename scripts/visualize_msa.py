"""
Generate a sequence logo at key promoted positions from the SadA MSA.

Run from scripts/:
    python visualize_msa.py
"""

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logomaker
from Bio import SeqIO

matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["axes.facecolor"]   = "white"
matplotlib.rcParams["svg.fonttype"]     = "none"
matplotlib.rcParams["font.family"]      = "sans-serif"
matplotlib.rcParams["font.sans-serif"]  = ["Helvetica", "Arial", "DejaVu Sans"]



SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
MSA_PATH = SCRIPT_DIR / ".." / "data" / "msa" / "sadA_full_clean.fasta"
OUTPUT_DIR = SCRIPT_DIR / ".." / "output" / "msa_visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMOTED = {
    38:  [("I", "V", "3-VRL"), ("V", "I", "4-IC")],
    48:  [("R", "C", "4-IC")],
    71:  [("I", "V", "1-VH")],
    152: [("F", "L", "2-L")],
    157: [("D", "G", "SadX")],
    172: [("R", "H", "1-VH")],
    233: [("Q", "R", "3-VRL")],
    261: [("F", "L", "3-VRL")],
}

SELECTED_POSITIONS = sorted(PROMOTED.keys())

AA_COLOR_SCHEME = {
    'G': 'red', 'A': 'red', 'V': 'red', 'C': 'red', 'P': 'red',
    'L': 'red', 'I': 'red', 'M': 'red', 'W': 'red', 'F': 'red',
    'S': 'green', 'T': 'green', 'Y': 'green', 'N': 'green', 'Q': 'green',
    'K': 'blue', 'R': 'blue', 'H': 'blue', 'D': 'blue', 'E': 'blue',
}

plt.rcParams.update({
    # "font.family": "serif",
    # "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 18,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def main():
    records = list(SeqIO.parse(MSA_PATH, "fasta"))
    sequences = [str(r.seq) for r in records]
    print(f"Loaded {len(sequences)} sequences from {MSA_PATH.name}")

    counts_df = logomaker.alignment_to_matrix(
        sequences, to_type="counts", characters_to_ignore="-"
    )
    freq_df = counts_df.div(counts_df.sum(axis=1), axis=0)
    freq_df = freq_df.iloc[[p - 1 for p in SELECTED_POSITIONS], :]
    freq_df = freq_df.reset_index(drop=True)

    logo = logomaker.Logo(
        freq_df,
        figsize=(8, 4),
        color_scheme=AA_COLOR_SCHEME,
        shade_below=0.5,
        fade_below=0.5,
    )

    # Position numbers only as tick labels
    logo.ax.set_xticks(range(len(SELECTED_POSITIONS)))
    logo.ax.set_xticklabels(
        [str(p) for p in SELECTED_POSITIONS],
        fontsize=13, fontweight="bold",
    )
    logo.ax.tick_params(axis="x", pad=6, length=4)

    # Annotations below: mutation on line 1, parent on line 2
    # For pos 38 with two mutations, join with " / "
    for i, pos in enumerate(SELECTED_POSITIONS):
        muts = PROMOTED[pos]
        mut_str = " / ".join(f"{f}\u2192{t}" for f, t, _ in muts)
        par_str = " / ".join(p for _, _, p in muts)

        logo.ax.annotate(
            mut_str,
            xy=(i, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -30), textcoords="offset points",
            ha="center", va="top",
            fontsize=11, color="#333",
        )
        logo.ax.annotate(
            par_str,
            xy=(i, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -44), textcoords="offset points",
            ha="center", va="top",
            fontsize=10, fontstyle="italic", color="#555",
        )

    logo.ax.set_ylabel("Frequency per Position")
    logo.ax.set_ylim(0, 1.0)
    logo.ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

    for patch in logo.ax.patches:
        patch.set_alpha(0.5)

    logo.fig.subplots_adjust(bottom=0.28)

    png_path = OUTPUT_DIR / "msa_logo.png"
    svg_path = OUTPUT_DIR / "msa_logo.svg"
    logo.fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    logo.fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()