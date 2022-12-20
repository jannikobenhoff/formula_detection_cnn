from PIL import Image

def open_image(path):
    im = Image.open(path)
    im.show()
