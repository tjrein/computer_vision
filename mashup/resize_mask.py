from PIL import Image
import numpy as np
import imageio

height = 200
width = 200

def load_image(filename):
    image = Image.open(filename).resize((height, width))
    return np.float32(image)

resize_mask = load_image("bojack_mask3.jpg")

imageio.imwrite("./bojack_mask3_resize.jpg", resize_mask)
