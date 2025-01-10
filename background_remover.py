import cv2
import numpy as np
from rembg import remove
from PIL import Image

def remove_background(image_path: str, output_path: str):
    """ Remove background from an image and save the output image.

    Args:
        image_path (str):
        output_path (str):
    """
    # Read image using PIL
    input_image = Image.open(image_path)

    # Remove background
    output_image = remove(input_image)

    # Save the output image for debugging purposes, no need to use the np.array() function
    output_image.save(output_path)