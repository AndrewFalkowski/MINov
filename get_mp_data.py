# Script for downloading Li containing materials from the Materials Project database.

import pandas as pd
from mp_api.client import MPRester
from tqdm import trange
import os

os.makedirs("MP_Li_Dataset", exist_ok=True)

with open("PATH TO YOUR API KEY", "r") as file:
    key = file.read().strip()


mpid = []
formula = []
structure = []
with MPRester(key) as mpr:
    for i in range(10):
        docs = mpr.materials.summary.search(
            chemsys=f"Li" + "-*" * i,  # add variable number of "-*"
            theoretical=False,
            fields=["material_id", "formula_pretty", "structure"],
        )
        mpid.extend([doc.material_id for doc in docs])
        formula.extend([doc.formula_pretty for doc in docs])
        structure.extend([doc.structure for doc in docs])

df = pd.DataFrame({"mpid": mpid, "formula": formula})
df.to_csv("mp_data.csv", index=False)

for idx in trange(len(formula)):
    structure[idx].to(fmt="cif", filename=f"MP_Li/{formula[idx]}.cif")

print("Done!")
