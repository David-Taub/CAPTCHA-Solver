import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


def create_image_char(image_size, background, character, char_color, char_pos, char_font):
    image = Image.new("RGBA", image_size, background)
    draw = ImageDraw.Draw(image)
    draw.text(char_pos, character, fill=char_color, font=char_font)
    return image


def generate_char_image(character, image_size=(100, 100), background_color=(255, 255, 255),
                        text_color=(0, 0, 0), font_path=r'fonts\FreeMono.ttf', font_size=100,
                        start_margins=(0, 0)):
    font = ImageFont.truetype(font_path, font_size)
    image = create_image_char(image_size, background_color, character,
                              text_color, start_margins, font)
    # H x W x C
    return image


def generate_multi_char_image(characters, image_size=[100, 600], background_color=(255, 255, 255)):
    imgs = [generate_char_image(character=c) for c in characters]
    arr = np.tile([[background_color]], (image_size + [1]))
    background_img = Image.fromarray(arr.astype(np.uint8))
    location = 20
    for img in imgs:
        background_img.paste(img, [location, 0])
        location += img.size[0]
    return background_img


def main():
    image = generate_multi_char_image('ABCDEF')
    # image = generate_char_image('A')
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
