import sys
import json
import os
import csv
import tqdm
import shutil

import generate_captcha


def create_dataset(dir_path, size, generator):
    id_length = len(str(size))

    shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path)
    with open(os.path.join(dir_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'alphabet': generator.alphabet,
                   'image_width': generator.image_size[1],
                   'image_height': generator.image_size[0],
                   'text_length': generator.length}, f)
    with open(os.path.join(dir_path, 'strings.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for i in tqdm.tqdm(range(size), desc=dir_path):
            sample_id = str(i).zfill(id_length)
            img, text = generator.generate_captcha()
            img.save(os.path.join(dir_path, f'{sample_id}.jpg'))
            writer.writerow([sample_id, text])


def main():
    for i, generator in enumerate(generate_captcha.generators):
        create_dataset(rf'data\level_{i}\train', int(sys.argv[1]), generator)
        create_dataset(rf'data\\level_{i}\test', 100, generator)


if __name__ == '__main__':
    main()
