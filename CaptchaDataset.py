import os
import csv
from torch.utils.data import Dataset
from PIL import Image


class CaptchaDataset(Dataset):
    def __init__(self, dir_path):
        self.dir_path = dir_path
        csvfile_path = os.path.join(dir_path, 'strings.csv')
        with open(csvfile_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            self.sample_ids, self.strings = zip(*reader)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.dir_path, f'{sample_id}.jpg')
        return Image.open(img_path), self.strings[idx]
