from pathlib import Path
import pickle

start_folder = Path("results")  # your starting folder

for pkl_file in start_folder.rglob("*.pkl"):

    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    except:
        print(f'\nEmpty file {pkl_file}')
        continue

    print(f'\n{pkl_file}')
    print(f'  File: {data.get("File Path", "")}')
    print(f'  Comment: {data.get("training dataset circumstance", "")}')
