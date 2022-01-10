import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchaudio


class WavDataset(Dataset):
    def __init__(self, root):
        self.folder = Path(root)
        self.files = glob.glob(root / "*.wav")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return torchaudio.load(self.files[idx]), torch.tensor(0)
