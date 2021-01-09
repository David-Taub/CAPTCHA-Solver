import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


def create_image_char(image_size, background, character, char_color, char_pos, char_font):
    image = Image.new("RGBA", image_size, background)
    draw = ImageDraw.Draw(image)
    draw.text(char_pos, character, fill=char_color, font=char_font)
    return image


def generate_text_image(text='ABCDab', image_size=(400, 100), background_color=(255, 255, 255), text_color=(0, 0, 0),
                        font_path=r'fonts\FreeMono.ttf', font_size=100, start_margins=(15, 0)):
    font = ImageFont.truetype(font_path, font_size)
    image = create_image_char(image_size, background_color, text,
                              text_color, start_margins, font)
    return image


def main():
    image = generate_text_image()
    plt.imshow(image)
    plt.show()


if __name__ == '__main__':
    main()
