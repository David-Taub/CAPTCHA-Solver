import random
import string

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


# TODO: random size, rotation, font, spacing, length, position, colors
# TODO: non-linear transformations
# TODO: random noise
# TODO: background patterns
# TODO: add lines and shapes


class CaptchGenerator:
    def reset_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)

    def __init__(self,
                 alphabet=string.ascii_lowercase, image_size=(100, 600), length_range=(6, 7),
                 offset_range=((0, 1), (0, 1)), rotation_range=(-60, 60), noise_sigma_range=(0, 100),
                 start_offset=20, background_color=(255, 255, 255, 255),
                 char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size_range=(100, 101),
                 start_margins=(0, 0)):
        self.alphabet = alphabet
        self.image_size = image_size
        self.length_range = length_range
        self.offset_range = offset_range
        self.start_offset = start_offset
        self.rotation_range = rotation_range
        self.noise_sigma_range = noise_sigma_range
        self.background_color = background_color
        self.char_color = char_color
        self.font_path = font_path
        self.font_size_range = font_size_range
        self.start_margins = start_margins
        self.reset_seed()

    def _generate_char_image(self, character):
        font_size = random.randrange(*self.font_size_range)
        font_obj = ImageFont.truetype(self.font_path, font_size)
        image_height = sum(font_obj.getmetrics())
        image_width = font_obj.font.getsize(character)[0][0]
        image = Image.new("RGBA", (image_width, image_height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, fill=self.char_color, font=font_obj)
        # H x W x C
        return image

    def _add_noise(self, img, sigma=100):
        img_arr = np.array(img)
        noise = np.random.normal(0, sigma, img_arr.shape)
        noised_arr = np.minimum(np.maximum(img_arr + noise, 0), 255)
        return Image.fromarray(noised_arr.astype(np.uint8))

    def _generate_multi_char_image(self, characters):
        imgs = [self._generate_char_image(character=c) for c in characters]
        background_array = np.tile([[self.background_color]], (list(self.image_size) + [1]))
        captcha_image = Image.fromarray(background_array.astype(np.uint8))

        x_location = self.start_offset
        for img in imgs:
            rotation_angle = random.randrange(*self.rotation_range)
            img_width = img.size[0]
            img = img.rotate(rotation_angle, resample=Image.BICUBIC, expand=True)
            base_y_offset = (captcha_image.size[1] - img.size[1]) / 2
            x_offset = random.randrange(*self.offset_range[0]) + img_width
            y_offset = int(base_y_offset + random.randrange(*self.offset_range[1]))
            if y_offset < 0:
                img = img.crop((0, -y_offset, img.size[0], img.size[1]))
                y_offset = 0
            captcha_image.alpha_composite(img, (x_location, y_offset))
            x_location += x_offset
        sigma = random.randrange(*self.noise_sigma_range)
        captcha_image = self._add_noise(captcha_image, sigma)
        return captcha_image.convert('RGB')

    def get_alphabet(self, with_capitals=True, with_numerics=True):
        alphabet = string.ascii_lowercase
        if with_capitals:
            alphabet += string.ascii_uppercase
        if with_numerics:
            alphabet += string.digits
        return alphabet

    def _generate_random_characters(self):
        length = random.randrange(*self.length_range)
        return ''.join(random.choice(self.alphabet) for x in range(length))

    def generate_captcha(self):
        characters = self._generate_random_characters()
        image = self._generate_multi_char_image(characters)
        return image, characters


# TODO: move to dataset creator
generators = [
    CaptchGenerator(alphabet=string.ascii_lowercase,
                    image_size=(100, 600), length_range=(6, 7), offset_range=((10, 11), (0, 1)),
                    rotation_range=(0, 1), noise_sigma_range=(0, 1),
                    start_offset=20, background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size_range=(90, 91),
                    start_margins=(0, 0)),
    CaptchGenerator(alphabet=string.ascii_lowercase + string.ascii_uppercase,
                    image_size=(100, 600), length_range=(6, 8), offset_range=((7, 13), (-5, 5)),
                    rotation_range=(-5, 5), noise_sigma_range=(0, 10),
                    start_offset=20, background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size_range=(90, 91),
                    start_margins=(0, 0)),
    CaptchGenerator(alphabet=string.ascii_lowercase + string.ascii_uppercase + string.digits,
                    image_size=(100, 600), length_range=(5, 10), offset_range=((-5, 10), (-15, 15)),
                    rotation_range=(-30, 30), noise_sigma_range=(5, 40),
                    start_offset=20, background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size_range=(90, 91),
                    start_margins=(0, 0)),
    CaptchGenerator(alphabet=string.ascii_lowercase + string.ascii_uppercase + string.digits,
                    image_size=(100, 600), length_range=(5, 11), offset_range=((-15, 0), (-20, 20)),
                    rotation_range=(-60, 60), noise_sigma_range=(30, 130),
                    start_offset=20, background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size_range=(70, 131),
                    start_margins=(0, 0)),
]


def main():
    N = 4
    M = 4
    for i in range(N):
        for j in range(M):
            plt.subplot(N, M, i * M + j + 1)
            plt.imshow(generators[-i - 1].generate_captcha()[0])
    plt.show()


if __name__ == '__main__':
    main()
