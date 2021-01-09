import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


def create_image_char(image_size, background, character, char_color, char_pos, char_font):
    image = Image.new("RGBA", image_size, background)
    draw = ImageDraw.Draw(image)
    draw.text(char_pos, character, fill=char_color, font=char_font)
    return image


def generate_char_image(character, image_size=(100, 100), background_color=(255, 255, 255, 255), text_color=(0, 0, 0, 255),
                        font_path=r'fonts\FreeMono.ttf', font_size=100, start_margins=(0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    image = create_image_char(image_size, background_color, character,
                              text_color, start_margins, font)
    # W x H x C
    return np.array(image)


def generate_multi_char_image(characters, image_size=[100, 400], background_color=(255, 255, 255, 255)):
    imgs = [generate_char_image(character=c) for c in characters]
    background_img = np.tile([[background_color]], (image_size + [1]))
    location = 0
    for img in imgs:
        background_img[:, location:location + img.shape[1], :] = img
        location += img.shape[1]
    return background_img


def main():
    image = generate_multi_char_image('ABC')
    # image = generate_char_image('A')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
