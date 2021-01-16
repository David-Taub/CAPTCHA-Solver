import json
import numpy as np
import os
import csv
from torch.utils.data import Dataset
from PIL import Image


class CaptchaDataset(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        with open(os.path.join(dir_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.alphabet_dict = dict([(v, i) for i, v in enumerate(self.metadata['alphabet'])])
        self.text_length = self.metadata['text_length']
        self.image_width = self.metadata['image_width']
        self.image_height = self.metadata['image_height']
        with open(os.path.join(dir_path, 'strings.csv'), 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.sample_ids, self.strings = zip(*reader)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.dir_path, f'{sample_id}.jpg')
        img = np.array(Image.open(img_path))
        img = ((np.transpose(img, (2, 0, 1)) / 255.0) - 0.5) * 2.0
        if self.transform:
            img = self.transform(img)
        string = self.strings[idx]
        numerized_string = np.array([self.alphabet_dict[c] for c in string])
        return img, numerized_string
