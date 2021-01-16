import random
import string

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw

random.seed(42)

# TODO: random size, rotation, font, spacing, length, position, colors
# TODO: non-linear transformations
# TODO: random noise
# TODO: background patterns
# TODO: add lines and shapes


class CaptchGenerator:
    def __init__(self,
                 alphabet=string.ascii_lowercase, image_size=(100, 600), length=6, offset_range=(0, 1),
                 start_offset=20, char_image_size=(100, 100), background_color=(255, 255, 255, 255),
                 char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size=100, start_margins=(0, 0)):
        self.alphabet = alphabet
        self.image_size = image_size
        self.length = length
        self.offset_range = offset_range
        self.start_offset = start_offset
        self.char_image_size = char_image_size
        self.background_color = background_color
        self.char_color = char_color
        self.font_path = font_path
        self.font_size = font_size
        self.start_margins = start_margins

    def _generate_char_image(self, character):
        font_obj = ImageFont.truetype(self.font_path, self.font_size)
        image = Image.new("RGBA", self.char_image_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, fill=self.char_color, font=font_obj)
        # H x W x C
        return image

    def _generate_multi_char_image(self, characters):
        imgs = [self._generate_char_image(character=c) for c in characters]
        blank_background = np.tile([[self.background_color]], (list(self.image_size) + [1]))
        background_img = Image.fromarray(blank_background.astype(np.uint8))
        location = self.start_offset
        for img in imgs:
            offset = random.randrange(*self.offset_range)
            background_img.alpha_composite(img, (location, 0))
            location += img.size[0] + offset
        return background_img.convert('RGB')

    def get_alphabet(self, with_capitals=True, with_numerics=True):
        alphabet = string.ascii_lowercase
        if with_capitals:
            alphabet += string.ascii_uppercase
        if with_numerics:
            alphabet += string.digits
        return alphabet

    def _generate_random_characters(self):
        return ''.join(random.choice(self.alphabet) for x in range(self.length))

    def generate_captcha(self):
        characters = self._generate_random_characters()
        image = self._generate_multi_char_image(characters)
        # image = np.array(image)
        return image, characters


# TODO: move to dataset creator
generators = [
    CaptchGenerator(alphabet=string.ascii_lowercase, image_size=(100, 600), length=6, offset_range=(0, 1),
                    start_offset=20, char_image_size=(100, 100), background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size=100, start_margins=(0, 0)),
    CaptchGenerator(alphabet=string.ascii_lowercase + string.ascii_uppercase + string.digits,
                    image_size=(100, 600), length=6, offset_range=(0, 1),
                    start_offset=20, char_image_size=(100, 100), background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size=100, start_margins=(0, 0)),
    CaptchGenerator(alphabet=string.ascii_lowercase + string.ascii_uppercase + string.digits,
                    image_size=(100, 600), length=6, offset_range=(-40, 0),
                    start_offset=20, char_image_size=(100, 100), background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size=100, start_margins=(0, 0)),
    CaptchGenerator(alphabet=string.ascii_lowercase + string.ascii_uppercase + string.digits,
                    image_size=(100, 600), length=6, offset_range=(-80, -30),
                    start_offset=20, char_image_size=(100, 100), background_color=(255, 255, 255, 255),
                    char_color=(0, 0, 0, 255), font_path=r'.\fonts\FreeMono.ttf', font_size=100, start_margins=(0, 0)),
]


def main():
    plt.imshow(generators[-1].generate_captcha()[0])
    plt.show()


if __name__ == '__main__':
    main()
