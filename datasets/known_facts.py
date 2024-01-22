from pathlib import Path
import os
import json

import torch
from torch.utils.data import Dataset


# This script downloads the 1000 KNOWN FACTS dataset as created 
# originally to ensure reproducibility

remote_url = "https://rome.baulab.info/data/dsets/known_1000.json"

class KnownsDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        data_dir = Path(data_dir)
        known_loc = data_dir / "known_1000.json"

        if not known_loc.exists():
            print(f"{known_loc} does not exist. Downloading from {remote_url}.")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(remote_url, known_loc)

        with open(known_loc, "r") as f:
            self.data = json.load(f)

        print(f"Loaded Knowns Dataset with {len(self.data)} elements.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]
    
if __name__ == "__main__":
    dataset = KnownsDataset(os.path.join(os.path.dirname(__file__), "../data"))