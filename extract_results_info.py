from pathlib import Path
import pickle

start_folder = Path("results")  # your starting folder

for pkl_file in start_folder.rglob("*.pkl"):
    with open(pkl_file, "rb") as f:
        print("")
        try:
            data = pickle.load(f)
            # do something with data
            print(f"{pkl_file}")
            print("  " + data.get("File Path", ""))
            print("  " + data.get("training dataset circumstance", ""))
        except:
            print(f'Empty file {pkl_file}')
