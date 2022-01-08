import json
from pathlib import Path

import torch.utils.data

from data.image import MyImageFolderDataset

GENDERS_ID = {
    'male': 0,
    'female': 1,
    'unknown': 2
}

class FFHQDataset(MyImageFolderDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(*args, **kwargs, root=str(Path(root) / 'images*'))
        self.root = Path(root)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        file_id = data['path'].split('/')[-1].split('.')[0]
        try:
            meta = json.load((self.root / 'ffhq-features-dataset' / 'json' / f'{file_id}.json').open())
            gender = meta[0]["faceAttributes"]["gender"]
        except:
            gender = 'unknown'
        return data["image"], torch.tensor(GENDERS_ID[gender])

