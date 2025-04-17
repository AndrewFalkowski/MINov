import os
import re
import glob
from pymatgen.core import Structure
from mp_api.client import MPRester

# Load your API key
with open("/Users/andrewf/Desktop/APIKEY.txt", "r") as file:
    API_KEY = file.read().strip()

# Directory containing your structure files
input_directory = "control_rex"  # Update this to your directory path
file_pattern = "*.cif"  # Update if you have different file extensions

# Get list of all cif files in the directory
cif_files = glob.glob(os.path.join(input_directory, file_pattern))

# Store problematic materials
not_found_materials = []

# Process each file
for cif_path in cif_files:
    try:
        # Extract the base filename without extension
        base_filename = os.path.splitext(os.path.basename(cif_path))[0]

        # Read the file content
        with open(cif_path, "r") as file:
            file_content = file.read()

        # Check if it's from Pearson database
        pearson_id_match = re.search(r"_database_code_PCD\s+(\d+)", file_content)
        if not pearson_id_match:
            # Also try alternative format seen in your example
            pearson_id_match = re.search(r"# End of data set (\d+)", file_content)

        if pearson_id_match:
            # It's a Pearson database file
            pearson_id = pearson_id_match.group(1)
            new_filename = f"{base_filename}_{pearson_id}.cif"
            new_path = os.path.join(input_directory, new_filename)

            # Rename the file
            os.rename(cif_path, new_path)
            print(f"Renamed Pearson file: {base_filename} → {new_filename}")

        else:
            # Not from Pearson, search Materials Project
            try:
                # Load the structure
                struct = Structure.from_file(cif_path)
                formula = struct.formula
                formula_elements = re.findall(r"([A-Z][a-z]*)", formula)
                spacegroup_symbol, spacegroup_number = struct.get_space_group_info()
                volume = struct.volume

                # Calculate volume tolerances
                vol_min = volume * 0.9
                vol_max = volume * 1.1

                # Query Materials Project database
                print(
                    f"Searching MP for: {base_filename} ({formula}, SG: {spacegroup_number})"
                )
                with MPRester(API_KEY) as mpr:
                    docs = mpr.materials.summary.search(
                        elements=formula_elements,
                        volume=[vol_min, vol_max],
                        spacegroup_number=spacegroup_number,
                        fields=["material_id", "formula_pretty"],
                    )

                # If not found, try with a wider volume tolerance
                if not docs:
                    vol_min = volume * 0.7
                    vol_max = volume * 1.3
                    with MPRester(API_KEY) as mpr:
                        docs = mpr.materials.summary.search(
                            elements=formula_elements,
                            volume=[vol_min, vol_max],
                            spacegroup_number=spacegroup_number,
                            fields=["material_id", "formula_pretty"],
                        )

                if docs:
                    # Get the best match (first result)
                    match = docs[0]
                    mp_id = match.material_id

                    # Create new filename with MP ID
                    new_filename = f"{base_filename}_{mp_id}.cif"
                    new_path = os.path.join(input_directory, new_filename)

                    # Rename the file
                    os.rename(cif_path, new_path)
                    print(f"Renamed MP file: {base_filename} → {new_filename}")
                else:
                    print(f"No MP match found for {base_filename}")
                    not_found_materials.append(base_filename)

            except Exception as e:
                print(f"Error processing {base_filename}: {str(e)}")
                not_found_materials.append(base_filename)

    except Exception as e:
        print(f"Error with file {cif_path}: {str(e)}")

# Print summary of problematic materials
if not_found_materials:
    print("\nMaterials with no MP match found:")
    for i, material in enumerate(not_found_materials, 1):
        print(f"{i}. {material}")
else:
    print("\nAll materials were successfully processed.")

print("Processing complete!")
