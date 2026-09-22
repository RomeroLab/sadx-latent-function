import pandas as pd

import pathlib

# # run in root repo
pathlib.Path("data/scratch/r64120_20221013_213137/3_C01").mkdir(parents=True, exist_ok=True)

samples = pd.read_csv("data/Oct22/Samples.csv")
reconstructed = samples[["barcode", "sample"]].rename(
    columns={"barcode": "Barcode", "sample": "Bio_Sample"}
)

with open("data/scratch/r64120_20221013_213137/3_C01/00Samples", "w") as f:
    f.write("# Reconstructed from Samples.csv\n")  # dummy first line (gets skipped)
    reconstructed.to_csv(f, sep="\t", index=False)