import cv2
import numpy as np
import color_filtering_v2 as cf
import get_countours_v2 as gc

colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "red": (0, 0, 255),
    "yellow": (0, 255, 255),
    "purple": (255, 0, 255),
    "orange": (0, 165, 255),
    "brown": (42, 42, 165),
    "pink": (147, 20, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "teal": (128, 128, 0),
    "lavender": (250, 230, 230),
}

log = 'centers.txt'

def cut_court(frame: np.ndarray) -> np.ndarray:
    """
    Only keep the court area in the frame.
    It is assumed to be a trapzoid shape on the bottm of the frame.
    Mark the trapzoid area.
    In shape(x, y), x is the width of the frame, y is the height of the frame, counting from the top.
    """
    # Define the vertices of the trapezoid
    bottom_left = (0, frame.shape[0]*0.8)
    top_left = (0.2*frame.shape[1], int(frame.shape[0] * 0.5))
    top_right = (frame.shape[1]*0.8, int(frame.shape[0] * 0.5))
    bottom_right = (frame.shape[1], frame.shape[0]*0.8)

    # Create a mask for the trapezoid area
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(
        [[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32
    )
    cv2.fillPoly(mask, pts, 255)

    # Apply the mask to the frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Draw the trapezoid on the frame
    # cv2.polylines(masked_frame, pts, isClosed=True, color=(255, 255, 255), thickness=2)

    return masked_frame

def log_out_top_contender_with_x_y_position(top_contour, frame_idx, file_path):
    with open(file_path, "a") as f:
        f.write(f"{frame_idx},{top_contour.center[0]},{top_contour.center[1]}\n")


def proccess_frame(frame_path: str, backgroundless_ball_path: str, output_path: str, debug: bool = False, tracker: gc.ContourTracker = None):
    filtered_frame = cf.create_filtered_frame(
        match_image_path=frame_path,
        ball_image_path=backgroundless_ball_path,
        output_path=f"{output_path}_filtered_debug.jpg",
        debug=debug,
    )
    if tracker is None:
        print("Error: tracker is None")
        exit(1)

    filtered_frame = cut_court(filtered_frame)
    # turn the frame into grayscale
    filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    
    kernel = np.ones((8, 8), np.uint8)
    filtered_frame = cv2.morphologyEx(filtered_frame, cv2.MORPH_CLOSE, kernel)
    
    contours = gc.get_contours(filtered_frame, frame_path)

    # save filtered frame for debugging
    if debug:
        filtered_path = f"{output_path}_filtered.jpg"
        cv2.imwrite(filtered_path, filtered_frame)

    top_cotours = tracker.find_top_candidates(contours, top_n=10)
    frame = cf.load_image(frame_path)
    
    if len(top_cotours) != 0:
    # print top contours, each with a different color
    log_out_top_contender_with_x_y_position(top_cotours[0], frame_path, log)


    # print top contours, each with a different color

    for i, top_contour in enumerate(top_cotours[:3]):
        print(top_contour)
        color = colors[list(colors.keys())[i]]
        # Use the origin frame, mark the top contours with a red circle, and the score above the circle
        cv2.circle(frame, top_contour.center, int(top_contour.radius), color, 2)
        cv2.putText(frame, str(top_contour.score), top_contour.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(output_path, frame)

    print(f"Top contours marked in {output_path}")
