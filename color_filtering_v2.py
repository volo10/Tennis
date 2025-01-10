import cv2
import numpy as np
from typing import Tuple
from utils import GameBall


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the specified file path.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Loaded image as a NumPy array.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    return image


def convert_to_hsv(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to HSV color space.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Image in HSV color space.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


def create_color_mask(hsv_image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    """
    Create a mask for pixels within the specified HSV color range.

    Args:
        hsv_image (np.ndarray): Image in HSV color space.
        lower_bound (np.ndarray): Lower bound for HSV values.
        upper_bound (np.ndarray): Upper bound for HSV values.

    Returns:
        np.ndarray: Binary mask of the specified color range.
    """
    return cv2.inRange(hsv_image, lower_bound, upper_bound)


def calculate_hsv_range(
    hsv_image: np.ndarray,
    mask: np.ndarray,
    hue_tolerance: int = 10,
    sv_tolerance: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the HSV range of the object using the provided mask, with added tolerance.

    Args:
        hsv_image (np.ndarray): HSV image of the object.
        mask (np.ndarray): Binary mask of the object.
        hue_tolerance (int, optional): Tolerance to add to the hue range. Defaults to 10.
        sv_tolerance (int, optional): Tolerance to add to the saturation and value ranges. Defaults to 20.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper HSV bounds.
    """
    object_pixels = hsv_image[mask > 0]
    lower_hsv = np.maximum(np.min(object_pixels, axis=0) - [hue_tolerance, sv_tolerance, sv_tolerance], 0)
    upper_hsv = np.minimum(
        np.max(object_pixels, axis=0) + [hue_tolerance, sv_tolerance, sv_tolerance],
        [179, 255, 255],
    )
    return lower_hsv.astype(np.uint8), upper_hsv.astype(np.uint8)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to an image.

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Resulting image with the mask applied.
    """
    return cv2.bitwise_and(image, image, mask=mask)


def save_image(image: np.ndarray, output_path: str) -> None:
    """
    Save an image to the specified file path.

    Args:
        image (np.ndarray): Image to save.
        output_path (str): Path to save the image.
    """
    cv2.imwrite(output_path, image)


def create_ball_mask(
    ball_image_path: str, hue_tolerance: int = 10, sv_tolerance: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mask and HSV range for the tennis ball using the ball-only image.

    Args:
        ball_image_path (str): Path to the ball-only image.
        hue_tolerance (int, optional): Tolerance to add to the hue range. Defaults to 10.
        sv_tolerance (int, optional): Tolerance to add to the saturation and value ranges. Defaults to 20.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Lower and upper HSV bounds for the ball.
    """
    ball_image = load_image(ball_image_path)
    ball_hsv = convert_to_hsv(ball_image)

    # Define a lenient yellow-green HSV range
    yellow_green_lower = np.array([18, 90, 90], dtype=np.uint8)
    yellow_green_upper = np.array([55, 255, 255], dtype=np.uint8)

    # Create a mask for the ball-only image
    non_black_mask = create_color_mask(ball_hsv, yellow_green_lower, yellow_green_upper)

    # Calculate the HSV range of the ball
    return calculate_hsv_range(ball_hsv, non_black_mask, hue_tolerance, sv_tolerance)


def get_filtered_frame(match_image: np.ndarray, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> np.ndarray:
    """
    Filter the match image to isolate the tennis ball using the precomputed HSV range.

    Args:
        match_image (np.ndarray): Input match image.
        lower_hsv (np.ndarray): Lower HSV bound for the ball.
        upper_hsv (np.ndarray): Upper HSV bound for the ball.

    Returns:
        np.ndarray: Resulting image with the ball isolated.
    """
    match_hsv = convert_to_hsv(match_image)
    match_mask = create_color_mask(match_hsv, lower_hsv, upper_hsv)
    return apply_mask(match_image, match_mask)


def create_filtered_frame(ball_image_path: str, match_image_path: str, output_path: str, debug: bool) -> None:
    """
    Main function to isolate the tennis ball in a match image using the ball-only image for color detection.

    Args:
        ball_image_path (str): Path to the ball-only image.
        match_image_path (str): Path to the match image.
        output_path (str): Path to save the resulting image.
    """
    # Step 1: Create the ball mask and HSV range
    if GameBall.is_set:
        lower_hsv, upper_hsv = GameBall.ball_low_hsv, GameBall.ball_high_hsv
    else:
        lower_hsv, upper_hsv = create_ball_mask(ball_image_path)
        GameBall.ball_low_hsv, GameBall.ball_high_hsv = lower_hsv, upper_hsv

    # Step 2: Load the match image
    match_image = load_image(match_image_path)

    # Step 3: Get the filtered frame
    result = get_filtered_frame(match_image, lower_hsv, upper_hsv)

    # Step 4: Save the resulting image for debugging
    if debug:
        save_image(result, output_path)
    # print(f"Result saved to {output_path}")
    return result
