#!/usr/bin/env python3
"""
Heatmap visualization of Hbin probabilities for SadA/SadX mutations,
with secondary-structure cartoon and annotation dots above.

Run from the PARENT of scripts/:
    python scripts/plot_hbin_heatmap.py
"""

import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
try:
    from Bio.PDB import DSSP
    _HAS_DSSP = True
except ImportError:
    _HAS_DSSP = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
CHARS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
C2I = {c: i for i, c in enumerate(CHARS)}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif']  = ['Times New Roman']
plt.rcParams['font.size']   = 16

# ──────────────────────────────────────────────────────────
# Paths  (script lives in scripts/, cwd = parent directory)
# ──────────────────────────────────────────────────────────
TSV_PATH    = "output/ordinal_Oct22_models/saved_models/6eeae50e.Hbin_probs.tsv"
PDB_PATH    = "data/pdb/SadA_rosetta_2024_3_6_p.pdb"
SAVE_PATH   = "output/singles_inference_heatmap/Hbin_probs_heatmap.png"
LEGEND_PATH = "output/singles_inference_heatmap/Hbin_probs_heatmap_legend.png"

# ──────────────────────────────────────────────────────────
# Annotation positions (all 1-indexed, consistent with TSV)
# ──────────────────────────────────────────────────────────
ACTIVE_SITE          = [141, 155, 157, 179, 181, 246, 255, 257, 261]
PREVIOUSLY_OPTIMIZED = [38, 71, 152, 172, 233, 261]

POSITIONS_PER_ROW = 100
START_POS = 2   # first 1-indexed position (M at 1 is trimmed)
TOP_N     = 10  # number of top predictions to label on the heatmap

# ──────────────────────────────────────────────────────────
# 3VRL wildtype sequence (full, untrimmed, 273 residues)
# Mutations in the TSV are defined relative to THIS sequence.
# The PDB (SadA Rosetta model) differs at previously-optimised
# positions — do NOT use PDB for wildtype identity.
# ──────────────────────────────────────────────────────────
WILDTYPE_SEQ_3VRL = (
    "MQHTYPAQLMRFGTAARAEHMTIAAAIHALDADEADAVVMDIVPDGERDAWWDDEGFSSSPFTK"
    "NAHHAGVVATSVTLGQLQREQGDKLVSKAAEYFGIACRVNDGLRTTRFVRLFSDALDAKPLTIG"
    "HDYEVEFLLATRRVYEPFEAPFNLAPHCGDVSYGRDTVNWPLKHSFPRQLGGFLTIQGADNDAG"
    "MVMWDNRPESRAALDEMHAEYRETGAIAALERAAKIMLKPRPGQLTLFQSKNLHAIERCTSTRRTMLL"
    # ↑ intentionally broken across lines for readability; joined below
)
# Paste the verified full sequence in one shot to avoid line-break errors:
WILDTYPE_SEQ_3VRL = (
    "MQHTYPAQLMRFGTAARAEHMTIAAAIHALDADEADAVVMDIVPDGERDAWWDDEGFSSSPFTK"
    "NAHHAGVVATSVTLGQLQREQGDKLVSKAAEYFGIACRVNDGLRTTRFVRLFSDALDAKPLTIG"
    "HDYEVEFLLATRRVYEPFEAPFNLAPHCGDVSYGRDTVNWPLKHSFPRQLGGFLTIQGADNDAG"
    "MVMWDNRPESRAALDEMHAEYRETGAIAALERAAKIMLKPRPGQLTLFQSKNLHAIERCTSTRR"
    "TMGLLLIHTEDGWRMFD"
)

# ──────────────────────────────────────────────────────────
# Secondary-structure source
# ──────────────────────────────────────────────────────────
# Priority:
#   1. PyMOL-exported TSV  (matches what you see in the viewer)
#   2. DSSP via mkdssp     (if binary is installed)
#   3. Hardcoded fallback  (last resort)
#
# To generate the PyMOL TSV, run once in PyMOL:
#   load <your.pdb>
#   run scripts/export_pymol_ss.py
#   export_ss data/pdb/SadA_ss_assignments.tsv
# ──────────────────────────────────────────────────────────
SS_TSV_PATH = "data/pdb/SadA_ss_assignments.tsv"

# Hardcoded DSSP assignments (fallback only — will NOT match PyMOL exactly)
_SADA_SS_DSSP = {
    1: "-", 2: "-", 3: "-", 4: "E", 5: "E", 6: "E", 7: "E", 8: "E",
    9: "E", 10: "E", 11: "E", 12: "T", 13: "T", 14: "E", 15: "E",
    16: "E", 17: "E", 18: "E", 19: "E", 20: "E", 21: "E", 22: "E",
    23: "H", 24: "H", 25: "H", 26: "H", 27: "H", 28: "H", 29: "H",
    30: "H", 31: "H", 32: "T", 33: "T", 34: "S", 35: "-", 36: "S",
    37: "E", 38: "E", 39: "E", 40: "E", 41: "E", 42: "-", 43: "-",
    44: "-", 45: "T", 46: "T", 47: "T", 48: "H", 49: "H", 50: "H",
    51: "H", 52: "H", 53: "T", 54: "-", 55: "G", 56: "G", 57: "G",
    58: "S", 59: "P", 60: "S", 61: "S", 62: "-", 63: "-", 64: "S",
    65: "S", 66: "-", 67: "-", 68: "-", 69: "-", 70: "S", 71: "G",
    72: "G", 73: "G", 74: "-", 75: "-", 76: "-", 77: "-", 78: "H",
    79: "H", 80: "H", 81: "H", 82: "H", 83: "T", 84: "-", 85: "-",
    86: "H", 87: "H", 88: "H", 89: "H", 90: "H", 91: "H", 92: "H",
    93: "H", 94: "H", 95: "H", 96: "H", 97: "H", 98: "H", 99: "H",
    100: "H", 101: "H", 102: "H", 103: "H", 104: "H", 105: "H",
    106: "H", 107: "H", 108: "T", 109: "T", 110: "S", 111: "H",
    112: "H", 113: "H", 114: "H", 115: "H", 116: "H", 117: "H",
    118: "H", 119: "H", 120: "H", 121: "T", 122: "-", 123: "E",
    124: "E", 125: "-", 126: "E", 127: "E", 128: "-", 129: "T",
    130: "T", 131: "S", 132: "-", 133: "E", 134: "E", 135: "-",
    136: "-", 137: "S", 138: "-", 139: "E", 140: "E", 141: "E",
    142: "E", 143: "E", 144: "-", 145: "S", 146: "-", 147: "-",
    148: "-", 149: "T", 150: "T", 151: "T", 152: "S", 153: "-",
    154: "-", 155: "B", 156: "-", 157: "-", 158: "-", 159: "T",
    160: "T", 161: "-", 162: "-", 163: "S", 164: "-", 165: "T",
    166: "T", 167: "T", 168: "-", 169: "S", 170: "-", 171: "-",
    172: "-", 173: "-", 174: "-", 175: "T", 176: "T", 177: "-",
    178: "E", 179: "E", 180: "E", 181: "E", 182: "E", 183: "E",
    184: "E", 185: "E", 186: "-", 187: "-", 188: "T", 189: "T",
    190: "-", 191: "-", 192: "-", 193: "E", 194: "E", 195: "E",
    196: "E", 197: "S", 198: "-", 199: "-", 200: "-", 201: "-",
    202: "S", 203: "H", 204: "H", 205: "H", 206: "H", 207: "H",
    208: "H", 209: "H", 210: "H", 211: "H", 212: "H", 213: "H",
    214: "T", 215: "T", 216: "T", 217: "S", 218: "S", 219: "-",
    220: "G", 221: "G", 222: "G", 223: "G", 224: "G", 225: "-",
    226: "-", 227: "E", 228: "E", 229: "E", 230: "E", 231: "-",
    232: "-", 233: "-", 234: "T", 235: "T", 236: "E", 237: "E",
    238: "E", 239: "E", 240: "E", 241: "E", 242: "T", 243: "T",
    244: "S", 245: "-", 246: "E", 247: "E", 248: "E", 249: "P",
    250: "P", 251: "-", 252: "S", 253: "S", 254: "-", 255: "E",
    256: "E", 257: "E", 258: "E", 259: "E", 260: "E", 261: "E",
    262: "E", 263: "E", 264: "E", 265: "E", 266: "T", 267: "T",
    268: "E", 269: "E", 270: "E", 271: "E", 272: "E", 273: "-",
}


# ══════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════

def _normalize_dssp(ss_dict):
    """Convert DSSP 8-state codes to 3-state (H/S/L) matching PyMOL convention.
    DSSP H,G -> H (helix);  E,B -> S (strand);  everything else -> L (loop)."""
    mapping = {}
    for resnum, code in ss_dict.items():
        if code in ("H", "G"):
            mapping[resnum] = "H"
        elif code in ("E", "B"):
            mapping[resnum] = "S"
        else:
            mapping[resnum] = "L"
    return mapping


def _load_pymol_ss(tsv_path):
    """Read PyMOL-exported SS assignments.
    Returns {resnum(int): code} where code is H, S, or L."""
    df = pd.read_csv(tsv_path, sep="\t")
    return {int(row["resnum"]): row["ss"] for _, row in df.iterrows()}


def get_ss(pdb_path, ss_tsv_path=SS_TSV_PATH):
    """Return {resnum: ss_code} for secondary structure.

    SS source priority:
      1. PyMOL-exported TSV  (best — matches the viewer)
      2. DSSP via mkdssp     (if installed)
      3. Hardcoded DSSP dict (last resort, may differ from PyMOL)

    All returned dicts use 3-state codes: H (helix), S (strand), L (loop).

    NOTE: wildtype sequence comes from WILDTYPE_SEQ_3VRL, not the PDB.
    The PDB (SadA) differs at previously-optimised positions.
    """
    # 1. PyMOL TSV (already in H/S/L)
    if os.path.isfile(ss_tsv_path):
        ss_dict = _load_pymol_ss(ss_tsv_path)
        print(f"SS source: PyMOL export ({ss_tsv_path}), "
              f"{len(ss_dict)} residues")
        return ss_dict

    # 2. DSSP (needs normalization)
    if _HAS_DSSP:
        try:
            parser    = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)
            model     = structure[0]
            dssp      = DSSP(model, pdb_path, dssp="mkdssp")
            ss_raw    = {key[1][1]: dssp[key][2] for key in dssp.keys()}
            ss_dict   = _normalize_dssp(ss_raw)
            print("SS source: DSSP (mkdssp).  "
                  "NOTE: may not match PyMOL — run export_pymol_ss.py "
                  "for exact match.")
            return ss_dict
        except (FileNotFoundError, OSError, Exception) as e:
            print(f"DSSP failed ({e.__class__.__name__}).")

    # 3. Hardcoded fallback (needs normalization)
    print(f"SS source: hardcoded DSSP fallback.  "
          f"NOTE: may not match PyMOL — run export_pymol_ss.py "
          f"for exact match.")
    return _normalize_dssp(_SADA_SS_DSSP)


def parse_mutations(tsv_path):
    """Return (mut_dict, wt_from_tsv, df).
    mut_dict:    (position, mut_aa) -> Hbin_prob
    wt_from_tsv: {position -> wt_aa}  (extracted from mutation labels)
    """
    df = pd.read_csv(tsv_path, sep="\t")
    mut_dict    = {}
    wt_from_tsv = {}
    for _, row in df.iterrows():
        feat   = row["feature"]
        wt_aa  = feat[0]
        mut_aa = feat[-1]
        pos    = int(feat[1:-1])
        mut_dict[(pos, mut_aa)] = row["Hbin_probs"]
        # Store the WT from the label; if position seen before, verify consistent
        if pos in wt_from_tsv:
            assert wt_from_tsv[pos] == wt_aa, (
                f"Inconsistent WT in TSV at position {pos}: "
                f"previously saw '{wt_from_tsv[pos]}', now '{wt_aa}' "
                f"(from '{feat}')")
        else:
            wt_from_tsv[pos] = wt_aa
    return mut_dict, wt_from_tsv, df


def validate_wildtype(wildtype_seq, wt_from_tsv):
    """Assert that the PDB sequence matches the wildtype encoded in every
    TSV mutation label.  Raises AssertionError with a clear message on
    the first mismatch."""
    mismatches = []
    for pos, tsv_wt in sorted(wt_from_tsv.items()):
        pdb_wt = wildtype_seq[pos - 1]   # 1-indexed -> 0-indexed
        if pdb_wt != tsv_wt:
            mismatches.append(f"  pos {pos}: PDB='{pdb_wt}'  TSV='{tsv_wt}'")

    assert len(mismatches) == 0, (
        f"Wildtype mismatch between PDB sequence and TSV mutation labels!\n"
        f"Found {len(mismatches)} mismatches:\n" +
        "\n".join(mismatches[:20]) +
        ("\n  ... (truncated)" if len(mismatches) > 20 else "") +
        "\n\nCheck that the PDB and TSV refer to the same protein/numbering."
    )
    print(f"WT validation : PASSED ({len(wt_from_tsv)} positions checked)")


def get_top_n(df, n=TOP_N):
    """Return list of (rank, position, mut_aa) for the top-N mutations."""
    top = df.nlargest(n, "Hbin_probs")
    return [(rank, int(row["feature"][1:-1]), row["feature"][-1])
            for rank, (_, row) in enumerate(top.iterrows(), start=1)]

def build_matrix(mut_dict, wildtype_seq):
    """20 x num_positions matrix.  WT cells -> NaN.
    Raises AssertionError if any non-WT mutation is missing from the TSV."""
    num_pos = len(wildtype_seq) - 1
    matrix  = np.full((20, num_pos), np.nan)
    missing = []
    for col in range(num_pos):
        pos   = col + START_POS
        wt_aa = wildtype_seq[pos - 1]
        for ai, aa in enumerate(CHARS):
            if aa == wt_aa:
                matrix[ai, col] = np.nan
            else:
                if (pos, aa) not in mut_dict:
                    missing.append(f"  {wt_aa}{pos}{aa}")
                else:
                    matrix[ai, col] = mut_dict[(pos, aa)]

    assert len(missing) == 0, (
        f"Missing {len(missing)} expected mutations from TSV!\n"
        + "\n".join(missing[:30])
        + ("\n  ... (truncated)" if len(missing) > 30 else "")
        + "\n\nEvery non-WT substitution at every position must be present."
    )
    print(f"Matrix built   : {matrix.shape}, all {num_pos * 19} "
          f"non-WT mutations present")
    return matrix


# ──────────────────────────────────────────────────────────
# Secondary-structure drawing
# ──────────────────────────────────────────────────────────

def _simplify_ss(code):
    """Map normalized SS code to helix / strand / coil.
    After loading, all dicts use: H = helix, S = strand, L = loop."""
    if code == "H":   return "helix"
    if code == "S":   return "strand"
    return "coil"


def _get_ss_segments(ss_dict, pos_start, pos_end):
    segments, cur_type, cur_start = [], None, None
    for pos in range(pos_start, pos_end + 1):
        s = _simplify_ss(ss_dict.get(pos, "-"))
        if s != cur_type:
            if cur_type is not None:
                segments.append((cur_type, cur_start, pos - 1))
            cur_type, cur_start = s, pos
    if cur_type is not None:
        segments.append((cur_type, cur_start, pos_end))
    return segments


def draw_ss_cartoon(ax, ss_dict, pos_start, pos_end, chunk_first_pos,
                    positions_per_row):
    segments = _get_ss_segments(ss_dict, pos_start, pos_end)
    y_mid = 0.5

    for ss_type, seg_s, seg_e in segments:
        x_left  = seg_s - chunk_first_pos
        x_right = seg_e - chunk_first_pos + 1

        if ss_type == "strand":
            head_len = min(0.8, (x_right - x_left) * 0.35)
            body_end = x_right - head_len
            ax.fill_between([x_left, body_end],
                            [y_mid - 0.22]*2, [y_mid + 0.22]*2,
                            color="black", alpha=0.85, lw=0)
            ax.fill([body_end, x_right, body_end],
                    [y_mid - 0.38, y_mid, y_mid + 0.38],
                    color="black", alpha=0.85)
        elif ss_type == "helix":
            n_res  = seg_e - seg_s + 1
            x_pts  = np.linspace(x_left, x_right, max(int(n_res * 30), 80))
            y_pts  = y_mid + 0.28 * np.sin(2 * np.pi * (x_pts - x_left))
            ax.plot(x_pts, y_pts, color="black", lw=2.2)
        else:
            ax.plot([x_left, x_right], [y_mid, y_mid], color="black", lw=1)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, positions_per_row)
    ax.axis("off")


# ──────────────────────────────────────────────────────────
# Annotation dots — ALL on one line, black borders
# ──────────────────────────────────────────────────────────

def draw_annotation_dots(ax, pos_start, pos_end, chunk_first_pos,
                         positions_per_row, top_muts):
    y = 0.5

    # ── active site (blue circles, black edge) ──
    for pos in ACTIVE_SITE:
        if pos_start <= pos <= pos_end:
            x = (pos - chunk_first_pos) + 0.5
            ax.plot(x, y, "o", color="#2166ac", markersize=15,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=5)

    # ── previously optimised (green squares, black edge) ──
    for pos in PREVIOUSLY_OPTIMIZED:
        if pos_start <= pos <= pos_end:
            x = (pos - chunk_first_pos) + 0.5
            ax.plot(x, y, "s", color="#4daf4a", markersize=14,
                    markeredgecolor="black", markeredgewidth=1.2, zorder=6)

    # ── top-N predictions (red circles, black edge, white number) ──
    for rank, pos, _mut in top_muts:
        if pos_start <= pos <= pos_end:
            x = (pos - chunk_first_pos) + 0.5
            ax.plot(x, y, "o", color="#c0392b", markersize=18,
                    markeredgecolor="black", markeredgewidth=1.3, zorder=7)
            # ax.text(x, y, str(rank), ha="center", va="center",
            #         fontsize=9, fontweight="bold", color="white",
            #         zorder=8)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, positions_per_row)
    ax.axis("off")


# ══════════════════════════════════════════════════════════
# Main figure
# ══════════════════════════════════════════════════════════

def plot_hbin_heatmap(matrix, wildtype_seq, ss_dict, top_muts,
                      save_path, legend_path,
                      positions_per_row=POSITIONS_PER_ROW):

    num_pos    = matrix.shape[1]
    num_chunks = (num_pos + positions_per_row - 1) // positions_per_row

    cell_w    = 0.35
    fig_width = positions_per_row * cell_w

    # Height ratios: SS (1.4) + dots (1.0) + heatmap (10) + spacer (2.8)
    height_ratios = []
    for ci in range(num_chunks):
        height_ratios.extend([1.4, 1.0, 10])
        if ci < num_chunks - 1:
            height_ratios.append(2.8)

    fig_height = num_chunks * 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = gridspec.GridSpec(len(height_ratios), 1,
                            height_ratios=height_ratios,
                            hspace=0.08, figure=fig)

    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="#e8e8e8")

    top_lookup = {(pos, mut): rank for rank, pos, mut in top_muts}

    gs_row = 0
    for ci in range(num_chunks):
        col_s = ci * positions_per_row
        col_e = min(col_s + positions_per_row, num_pos)
        width = col_e - col_s

        chunk_first = START_POS + col_s
        chunk_last  = START_POS + col_e - 1
        chunk_mat   = matrix[:, col_s:col_e]

        # ── secondary structure ──
        ax_ss = fig.add_subplot(gs[gs_row]); gs_row += 1
        draw_ss_cartoon(ax_ss, ss_dict,
                        chunk_first, chunk_last,
                        chunk_first, positions_per_row)
        if ci == 0:
            ax_ss.set_title("Secondary structure", fontsize=32,
                            loc="left", pad=2, style="italic")

        # ── annotation dots ──
        ax_dot = fig.add_subplot(gs[gs_row]); gs_row += 1
        draw_annotation_dots(ax_dot, chunk_first, chunk_last,
                             chunk_first, positions_per_row, top_muts)

        # ── heatmap ──
        ax_hm = fig.add_subplot(gs[gs_row]); gs_row += 1

        sns.heatmap(chunk_mat, cmap=cmap, cbar=False,
                    yticklabels=CHARS, xticklabels=False,
                    ax=ax_hm,
                    linewidths=0.5, linecolor="white",
                    vmin=vmin, vmax=vmax)

        # Y-axis amino-acid labels
        ax_hm.set_yticklabels(ax_hm.get_yticklabels(),
                              fontsize=16, va="center", rotation=0)
        ax_hm.tick_params(axis="y", length=0, pad=4)

        # Wild-type markers
        for c in range(width):
            pos   = chunk_first + c
            wt_aa = wildtype_seq[pos - 1]
            if wt_aa in C2I:
                r = C2I[wt_aa]
                ax_hm.add_patch(
                    Rectangle((c, r), 1, 1, fill=False,
                              edgecolor="black", linewidth=1.3))
                ax_hm.plot(c + 0.5, r + 0.5, marker=".",
                           color="black", markersize=6)

        # ── overlay top-N rank numbers on heatmap cells ──
        for c in range(width):
            pos = chunk_first + c
            for ai, aa in enumerate(CHARS):
                rank = top_lookup.get((pos, aa))
                if rank is not None:
                    pass
                    # ax_hm.text(
                    #     c + 0.5, ai + 0.5, str(rank),
                    #     ha="center", va="center",
                    #     fontsize=10, fontweight="bold", color="white",
                    #     path_effects=[pe.withStroke(linewidth=3,
                    #                                 foreground="black")],
                    #     zorder=10)

        # X-axis labels (every 5th)
        ticks  = np.arange(width)
        labels = [str(chunk_first + t)
                  if (chunk_first + t) % 5 == 0 else ""
                  for t in ticks]
        ax_hm.set_xticks(ticks + 0.5)
        ax_hm.set_xticklabels(labels, rotation=45, ha="center", fontsize=14)
        ax_hm.set_ylabel("Amino Acid", fontsize=28)
        ax_hm.set_xlim(0, positions_per_row)
        ax_hm.set_xlabel(
            f"Position in SadA sequence ({chunk_first}\u2013{chunk_last})",
            fontsize=32)

        # spacer
        if ci < num_chunks - 1:
            ax_sp = fig.add_subplot(gs[gs_row]); gs_row += 1
            ax_sp.axis("off")

    # ── legend on last heatmap ──
    last_hm = [a for a in fig.axes
               if hasattr(a, 'get_xlabel') and 'Position' in a.get_xlabel()][-1]
    last_hm.plot([], [], "o", color="#c0392b", markersize=14,
                 markeredgecolor="black", markeredgewidth=1.2,
                 label="Top prediction (ranked)")
    last_hm.plot([], [], "o", color="#2166ac", markersize=14,
                 markeredgecolor="black", markeredgewidth=1.2,
                 label="Active site")
    last_hm.plot([], [], "s", color="#4daf4a", markersize=13,
                 markeredgecolor="black", markeredgewidth=1.2,
                 label="Previously optimised")
    last_hm.plot([], [], "s", color="#e8e8e8", markeredgecolor="black",
                 markersize=13, markeredgewidth=1.2,
                 label="Wild type (no score)")
    last_hm.legend(loc="lower right", fontsize=14, framealpha=0.95,
                   handletextpad=0.5, borderpad=0.6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap  -> {save_path}")

    # ── separate HORIZONTAL colour-bar with big title ──
    fig_cb, ax_cb = plt.subplots(figsize=(8, 1.8))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("High Bin Probability", fontsize=32, fontweight="bold",
                   labelpad=10)
    fig_cb.tight_layout()
    os.makedirs(os.path.dirname(legend_path) or ".", exist_ok=True)
    fig_cb.savefig(legend_path, dpi=300, bbox_inches="tight")
    plt.close(fig_cb)
    print(f"Legend   -> {legend_path}")


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":

    seq = WILDTYPE_SEQ_3VRL
    assert len(seq) == 273, f"3VRL sequence length is {len(seq)}, expected 273"

    ss_dict              = get_ss(PDB_PATH)
    mut_dict, wt_tsv, df = parse_mutations(TSV_PATH)

    # ── Validate that 3VRL sequence matches TSV wildtype labels ──
    validate_wildtype(seq, wt_tsv)

    top_muts = get_top_n(df, TOP_N)
    matrix   = build_matrix(mut_dict, seq)

    print(f"Sequence length : {len(seq)}")
    print(f"Matrix shape    : {matrix.shape}")
    print(f"Value range     : {np.nanmin(matrix):.4f} - {np.nanmax(matrix):.4f}")
    print(f"Active site     : {ACTIVE_SITE}")
    print(f"Previously opt. : {PREVIOUSLY_OPTIMIZED}")
    print(f"Top {TOP_N} predictions:")
    for rank, pos, mut in top_muts:
        wt = seq[pos - 1]
        print(f"  #{rank:>2d}  {wt}{pos}{mut}  ({mut_dict[(pos, mut)]:.4f})")

    plot_hbin_heatmap(matrix, seq, ss_dict, top_muts, SAVE_PATH, LEGEND_PATH)