"""
Run this INSIDE PyMOL to export secondary-structure assignments.

Usage (PyMOL command line):
    load data/pdb/SadA_rosetta_2024_3_6_p.pdb
    run scripts/export_pymol_ss.py
    export_ss data/pdb/SadA_ss_assignments.tsv

Or from the terminal:
    pymol data/pdb/SadA_rosetta_2024_3_6_p.pdb -cqr scripts/export_pymol_ss.py \
          -d "export_ss data/pdb/SadA_ss_assignments.tsv"

The output TSV has two columns: resnum and ss
    ss values:  H = helix,  S = strand,  L = loop/coil
"""

from pymol import cmd, stored


def export_ss(out_path="data/pdb/SadA_ss_assignments.tsv", selection="all"):
    """
    Export per-residue secondary structure from the currently loaded
    PyMOL session to a two-column TSV file.

    PyMOL SS codes:  H = helix, S = strand, L = loop/coil ('' = loop)
    """
    stored.ss_list = []
    cmd.iterate(
        f"({selection}) and name CA",
        "stored.ss_list.append((resi, ss))"
    )

    with open(out_path, "w") as f:
        f.write("resnum\tss\n")
        for resi, ss in stored.ss_list:
            # PyMOL uses '' (empty) for loops; normalise to 'L'
            ss_code = ss if ss in ("H", "S") else "L"
            f.write(f"{resi}\t{ss_code}\n")

    print(f"Wrote {len(stored.ss_list)} residues -> {out_path}")


# Register as a PyMOL command
cmd.extend("export_ss", export_ss)