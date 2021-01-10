import json
import os
import generate_captcha
import csv
import tqdm


def create_dataset(dir_path, size):
    ID_LENGTH = 8
    TEXT_LENGTH = 6
    IMAGE_WIDTH = 600
    IMAGE_HEIGHT = 100

    if os.path.isdir(dir_path) and len(os.listdir(dir_path)) == size + 1:
        return
    os.makedirs(dir_path)
    alphabet = generate_captcha.get_alphabet()
    with open(os.path.join(dir_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump({'alphabet': alphabet,
                   'image_width': IMAGE_WIDTH,
                   'image_height': IMAGE_HEIGHT,
                   'text_length': TEXT_LENGTH}, f)
    with open(os.path.join(dir_path, 'strings.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        for i in tqdm.tqdm(range(size)):
            sample_id = str(i).zfill(ID_LENGTH)
            img, text = generate_captcha.generate_captcha(alphabet, [IMAGE_HEIGHT, IMAGE_WIDTH], TEXT_LENGTH)
            img.save(os.path.join(dir_path, f'{sample_id}.jpg'))
            writer.writerow([sample_id, text])


def main():
    create_dataset(r'data\train', 1000)
    create_dataset(r'data\test', 1000)
    # import CaptchaDataset
    # d = CaptchaDataset.CaptchaDataset(r'data\train')
    # print(len(d))
    # print(d[4])
    # import matplotlib.pyplot as plt
    # plt.imshow(d[4][0])
    # plt.show()


if __name__ == '__main__':
    main()
