# This is a script to search bz2 files for a specific fighter string. This is what I used for searching for missing odds in investigative work.

import bz2
import os

# === CONFIGURATION ===
folder_path = download_folder  # <- CHANGE THIS
search_string = 'buday'     # <- CHANGE THIS

# === SEARCH FUNCTION ===
def search_bz2_files(folder, target):
    matches = []

    for filename in os.listdir(folder):
        if filename.endswith('.bz2'):
            full_path = os.path.join(folder, filename)
            try:
                with bz2.open(full_path, 'rt', encoding='utf-8', errors='ignore') as file:
                    for i, line in enumerate(file, 1):
                        if target.lower() in line.lower():
                            matches.append((filename, i, line.strip()))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return matches

# === RUN SEARCH AND PRINT RESULTS ===
results = search_bz2_files(folder_path, search_string)

print(f"\nFound {len(results)} match(es):\n")
for filename, line_number, line in results:
    print(f"[{filename}] Line {line_number}: {line}")
