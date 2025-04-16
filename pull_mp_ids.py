import os
import re
import glob
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from mp_api.client import MPRester
import pandas as pd

# Load your API key
with open("/Users/andrewf/Desktop/APIKEY.txt", "r") as file:
    key = file.read().strip()

# Directory containing your structure files
input_directory = "perovskite_rex"  # Update this to your directory path
file_pattern = "*.cif"  # Update if you have different file extensions

# Get list of all cif files in the directory
cif_files = glob.glob(os.path.join(input_directory, file_pattern))

# Store problematic materials
not_found_materials = []
high_energy_materials = []

# Define space group to crystal system mapping for special cases
crystal_system_prefixes = {
    221: "[cub]",  # Cubic
    140: "[tet]",  # Tetragonal
    62: "[ortho]",  # Orthorhombic
}

# Process each file
for cif_path in cif_files:
    try:
        # Extract the base filename
        base_filename = os.path.basename(cif_path)

        # Load the structure
        struct = Structure.from_file(cif_path)

        # Get structure information
        spacegroup_symbol, spacegroup_number = struct.get_space_group_info()
        volume = struct.volume
        formula = struct.formula
        formula_elements = re.findall(r"([A-Z][a-z]*)", formula)

        print(f"Processing: {base_filename}")
        print(f"  Formula: {formula}")
        print(f"  Space group: {spacegroup_number} ({spacegroup_symbol})")
        print(f"  Volume: {volume:.2f} Å³")

        # Try with original volume (with 10% tolerance)
        vol_min = volume * 0.9
        vol_max = volume * 1.1

        # Query Materials Project database with volume tolerance
        with MPRester(key) as mpr:
            docs = mpr.materials.summary.search(
                elements=formula_elements,
                volume=[vol_min, vol_max],
                spacegroup_number=spacegroup_number,
                fields=["material_id", "formula_pretty", "energy_above_hull", "volume"],
            )

        # If not found, try with half volume (primitive cell) with 10% tolerance
        if not docs:
            half_volume = volume / 2
            vol_min = half_volume * 0.9
            vol_max = half_volume * 1.1

            print(
                f"  No match found with original volume, trying half volume: {half_volume:.2f} Å³"
            )

            with MPRester(key) as mpr:
                docs = mpr.materials.summary.search(
                    elements=formula_elements,
                    volume=[vol_min, vol_max],
                    spacegroup_number=spacegroup_number,
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "energy_above_hull",
                        "volume",
                    ],
                )

        # Check if any matches were found
        if not docs:
            print(f"  WARNING: No MP match found for {base_filename}")
            not_found_materials.append(
                {
                    "filename": base_filename,
                    "formula": formula,
                    "space_group": spacegroup_number,
                    "volume": volume,
                }
            )
            continue

        # Get the best match (first result)
        match = docs[0]

        # Check energy above hull
        if match.energy_above_hull > 0.1:
            print(
                f"  WARNING: High energy_above_hull ({match.energy_above_hull:.3f} eV/atom) for {match.material_id}"
            )
            high_energy_materials.append(
                {
                    "filename": base_filename,
                    "formula": formula,
                    "mp_id": match.material_id,
                    "energy_above_hull": match.energy_above_hull,
                }
            )

        # Add crystal system prefix if spacegroup is in our special cases
        prefix = ""
        if spacegroup_number in crystal_system_prefixes:
            prefix = crystal_system_prefixes[spacegroup_number]

        # Create new filename with MP ID and appropriate prefix
        new_filename = f"{prefix}{match.formula_pretty}_{match.material_id}.cif"
        new_path = os.path.join(input_directory, new_filename)

        # Rename the file
        os.rename(cif_path, new_path)

        print(f"  Renamed to: {new_filename}")
        print(f"  MP ID: {match.material_id}")
        print(f"  MP Volume: {match.volume:.2f} Å³")
        print(f"  Energy above hull: {match.energy_above_hull:.3f} eV/atom")
        print("-" * 50)

    except Exception as e:
        print(f"Error processing {cif_path}: {str(e)}")
        not_found_materials.append(
            {"filename": os.path.basename(cif_path), "error": str(e)}
        )

# Print summary of problematic materials
print("\n" + "=" * 70)
print("SUMMARY OF PROBLEMATIC MATERIALS")
print("=" * 70)

if not_found_materials:
    print("\nMaterials with no MP match found:")
    for i, material in enumerate(not_found_materials, 1):
        print(f"{i}. {material['filename']}")
        if "formula" in material:
            print(f"   Formula: {material['formula']}")
            print(f"   Space Group: {material['space_group']}")
            print(f"   Volume: {material['volume']:.2f} Å³")
        if "error" in material:
            print(f"   Error: {material['error']}")
        print()
else:
    print("\nAll materials were successfully matched with MP database.")

if high_energy_materials:
    print("\nMaterials with high energy above hull (> 0.1 eV/atom):")
    for i, material in enumerate(high_energy_materials, 1):
        print(f"{i}. {material['filename']} → {material['mp_id']}")
        print(f"   Formula: {material['formula']}")
        print(f"   Energy above hull: {material['energy_above_hull']:.3f} eV/atom")
        print()
else:
    print("\nNo materials with high energy above hull were found.")

print("Processing complete!")
