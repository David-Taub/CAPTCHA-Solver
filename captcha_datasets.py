import json
import numpy as np
import os
import csv
from torch.utils.data import Dataset
from PIL import Image
import generate_captcha


class DynamicCaptchaDataset(Dataset):

    def __init__(self, generator: generate_captcha.CaptchGenerator,
                 transform=None, padded_text_length=10, fake_length=1000, shuffle=True):
        self.generator = generator
        self.shuffle = shuffle
        self.transform = transform
        self.padded_text_length = padded_text_length
        self.alphabet_dict = dict(
            [(v, i) for i, v in enumerate([''] + list(self.generator.alphabet))])
        self.fake_length = fake_length

    def __len__(self):
        return self.fake_length

    def __getitem__(self, idx):
        if idx == 0 and not self.shuffle:
            self.generator.reset_seed()
        img, string = self.generator.generate_captcha()
        if self.transform:
            img = self.transform(img)
        # pad
        string = list(string)
        string += [''] * (self.padded_text_length - len(string))
        numerized_string = np.array([self.alphabet_dict[c] for c in string])
        return img, numerized_string


class CaptchaDataset(Dataset):

    def __init__(self, dir_path, transform=None, padded_text_length=10):
        self.dir_path = dir_path
        self.transform = transform
        self.padded_text_length = padded_text_length
        with open(os.path.join(dir_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.alphabet_dict = dict(
            [(v, i) for i, v in enumerate([''] + list(self.metadata['alphabet']))])
        with open(os.path.join(dir_path, 'strings.csv'), 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            self.sample_ids, self.strings = zip(*reader)

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.dir_path, f'{sample_id}.jpg')
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        string = list(self.strings[idx])
        # pad
        string += [''] * (self.padded_text_length - len(string))
        numerized_string = np.array([self.alphabet_dict[c] for c in string])
        return img, numerized_string
