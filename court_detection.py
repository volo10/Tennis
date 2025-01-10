# import cv2
# import numpy as np
# import matplotlib.pyplot as plt


# def detect_and_mark_squares(image_path, output_path="output_image.png"):
#     # Load the image
#     img = cv2.imread(image_path)

#     if img is None:
#         print(f"Error: Unable to load image at {image_path}.")
#         return

#     # Convert to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Apply binary thresholding
#     _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
#     plt.imshow(binary)
#     # Detect edges for the net
#     edges = cv2.Canny(binary, 50, 150, apertureSize=3)

#     # Use Hough Line Transform to find the black line (bottom of the net)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=50)
#     net_bottom_y = None

#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             # Look for the lowest horizontal line (assuming the net is horizontal)
#             if abs(y1 - y2) < 10:  # Horizontal line check
#                 if net_bottom_y is None or y1 > net_bottom_y:
#                     net_bottom_y = y1

#     if net_bottom_y is None:
#         print("Error: Could not detect the bottom of the net.")
#         return

#     print(f"Bottom of the net detected at y = {net_bottom_y}")

#     # Create a list to store white pixels
#     white_pixels = []
#     _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
#     plt.imshow(binary)
#     # Scan the region below the net_bottom_y for white pixels
#     for y in range(net_bottom_y + 1, img.shape[0]):
#         for x in range(img.shape[1]):
#             if binary[y, x] == 255:  # Check for white pixel
#                 white_pixels.append((x, y))

#     print(f"Total white pixels detected below the net: {len(white_pixels)}")

#     # Create a new image to draw the white pixels in blue
#     blue_image = np.zeros_like(img)
#     for x, y in white_pixels:
#         blue_image[y, x] = [255, 0, 0]  # Blue color in BGR

#     # Save the new image with blue pixels
#     cv2.imwrite("blue_pixels_image.png", blue_image)
#     print("Image with blue pixels saved as 'blue_pixels_image.png'")

#     # Save the original image with the detected line
#     cv2.line(img, (0, net_bottom_y), (img.shape[1], net_bottom_y), (255, 255, 255), 2)

#     # Ensure the output path has a valid extension
#     if not output_path.lower().endswith((".jpg", ".jpeg", ".png")):
#         output_path += ".png"

#     cv2.imwrite(output_path, img)
#     print(f"Processed image with the net line saved to {output_path}")

#     # Save the result as an image instead of displaying it
#     plt.figure(figsize=(10, 10))
#     plt.subplot(1, 2, 1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Net Line")
#     plt.axis("off")

#     plt.subplot(1, 2, 2)
#     plt.imshow(cv2.cvtColor(blue_image, cv2.COLOR_BGR2RGB))
#     plt.title("White Pixels Below Net in Blue")
#     plt.axis("off")

#     plt.savefig("result_plot.png")
#     print("Result plot saved as 'result_plot.png'")


# # Path to your image and output
# image_path = "frames/frame0.jpg"
# output_path = "output_image_with_net.png"

# detect_and_mark_squares(image_path, output_path)

import cv2
import numpy as np
import matplotlib.pyplot as plt


def group_white_pixels_by_lines(binary_image_path, output_path="grouped_lines.png"):
    # Load the binary image
    binary_img = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(binary_img, 10, 255, cv2.THRESH_BINARY)
    if binary_img is None:
        print(f"Error: Unable to load image at {binary_image_path}.")
        return

    # Detect edges using Canny (to identify potential line candidates)
    edges = cv2.Canny(binary_img, 50, 150, apertureSize=3)

    # Use Probabilistic Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    if lines is None:
        print("Error: No lines detected in the image.")
        return

    print(f"Total lines detected: {len(lines)}")

    # Prepare a list of detected lines
    line_segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_segments.append(((x1, y1), (x2, y2)))

    # Function to calculate the distance of a point to a line segment
    def point_to_line_distance(point, line):
        (x0, y0) = point
        (x1, y1), (x2, y2) = line
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        return numerator / denominator if denominator != 0 else float("inf")

    # Filter lines to ensure minimum spacing of 5 pixels between vertical and horizontal lines
    def filter_lines_by_spacing(lines, orientation="vertical", min_spacing=5):
        filtered_lines = []
        sorted_lines = sorted(lines, key=lambda l: l[0][0] if orientation == "vertical" else l[0][1])
        prev_pos = -float("inf")

        for line in sorted_lines:
            pos = line[0][0] if orientation == "vertical" else line[0][1]
            if abs(pos - prev_pos) >= min_spacing:
                filtered_lines.append(line)
                prev_pos = pos
        return filtered_lines

    # Separate vertical and horizontal lines
    vertical_lines = []
    horizontal_lines = []

    for line in line_segments:
        (x1, y1), (x2, y2) = line
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if 40 <= angle <= 130:  # Near vertical
            vertical_lines.append(line)
        elif angle <= 10 or angle >= 170:  # Near horizontal
            horizontal_lines.append(line)

    # Filter lines by minimum spacing
    vertical_lines = filter_lines_by_spacing(vertical_lines, orientation="vertical", min_spacing=5)
    horizontal_lines = filter_lines_by_spacing(horizontal_lines, orientation="horizontal", min_spacing=5)

    print(f"Filtered vertical lines: {len(vertical_lines)}")
    print(f"Filtered horizontal lines: {len(horizontal_lines)}")

    # Group white pixels based on proximity to detected lines
    height, width = binary_img.shape
    grouped_pixels = [[] for _ in range(len(vertical_lines) + len(horizontal_lines))]

    for y in range(height):
        for x in range(width):
            if binary_img[y, x] == 255:  # Check for white pixel
                min_distance = float("inf")
                closest_line_idx = -1

                # Find the closest line among filtered lines
                all_lines = vertical_lines + horizontal_lines
                for idx, line in enumerate(all_lines):
                    distance = point_to_line_distance((x, y), line)
                    if distance < min_distance:
                        min_distance = distance
                        closest_line_idx = idx

                # Threshold to ensure pixel is close enough to be part of the line
                if min_distance < 7:  # 5-pixel threshold
                    grouped_pixels[closest_line_idx].append((x, y))

    # Create an output image
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255)]

    # Draw grouped pixels with unique colors for each line
    for idx, pixels in enumerate(grouped_pixels):
        color = colors[idx % len(colors)]
        for x, y in pixels:
            output_img[y, x] = color

        # Add line index text near the center of the line
        if pixels:
            center_x = sum(p[0] for p in pixels) // len(pixels)
            center_y = sum(p[1] for p in pixels) // len(pixels)
            cv2.putText(output_img, str(idx + 1), (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        print(f"Line {idx + 1}: {len(pixels)} pixels grouped")

    # Save and display the result
    cv2.imwrite(output_path, output_img)
    print(f"Output image with grouped pixels saved to {output_path}")

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.title("Grouped White Pixels by Lines")
    plt.axis("off")
    plt.show()


# Path to binary image
binary_image_path = "images/frame0_blackout.jpg"
output_path = "grouped_lines.png"
group_white_pixels_by_lines(binary_image_path, output_path)
