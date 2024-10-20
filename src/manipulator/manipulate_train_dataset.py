from PIL import Image, ImageDraw, ImageFont
import random


def add_text_to_image(image, text="Sample Text"):
    """
    Add text to an image, to simulate a manipulated image
    :param image:
    :param text:
    :return:
    """
    if isinstance(image, str):
        image = Image.open(image)

    if image.mode != "RGB":
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    width, height = image.size
    for _ in range(random.randint(3, 10)):
        font_size = random.randint(10, 50)
        font = ImageFont.truetype("arial.ttf", font_size)
        position = (random.randint(0, width - font_size), random.randint(0, height - font_size))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw.text(position, text, font=font, fill=color)

    return image
