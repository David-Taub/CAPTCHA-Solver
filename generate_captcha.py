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


def create_image_char(char_image_size, background, character, char_color, char_pos, char_font):
    image = Image.new("RGBA", char_image_size, background)
    draw = ImageDraw.Draw(image)
    draw.text(char_pos, character, fill=char_color, font=char_font)
    return image


def generate_char_image(character, char_image_size=(100, 100), background_color=(255, 255, 255),
                        text_color=(0, 0, 0), font_path=r'fonts\FreeMono.ttf', font_size=100,
                        start_margins=(0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    image = create_image_char(char_image_size, background_color, character,
                              text_color, start_margins, font)
    # H x W x C
    return image


def generate_multi_char_image(characters, image_size, background_color=(255, 255, 255)):
    imgs = [generate_char_image(character=c) for c in characters]
    arr = np.tile([[background_color]], (image_size + [1]))
    background_img = Image.fromarray(arr.astype(np.uint8))
    location = 20
    for img in imgs:
        background_img.paste(img, [location, 0])
        location += img.size[0]
    return background_img


def get_alphabet(with_capitals=True, with_numerics=True):
    alphabet = string.ascii_lowercase
    if with_capitals:
        alphabet += string.ascii_uppercase
    if with_numerics:
        alphabet += string.digits
    return alphabet


def rand_text_generator(alphabet, length):
    return ''.join(random.choice(alphabet) for x in range(length))


def generate_captcha(alphabet, image_size=[100, 600], length=6):
    text = rand_text_generator(alphabet, length)
    image = generate_multi_char_image(text, image_size)
    return image, text


def main():
    plt.imshow(generate_captcha())
    plt.show()


if __name__ == '__main__':
    main()
