import os
import generate_captcha
import csv
import tqdm


def create_dataset(dir_path, size):
    ID_LENGTH = 8
    if os.path.isdir(dir_path):
        return
    os.makedirs(dir_path)
    with open(os.path.join(dir_path, 'strings.csv'), 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in tqdm.tqdm(range(size)):
            sample_id = str(i).zfill(ID_LENGTH)
            img, text = generate_captcha.generate_captcha()
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
