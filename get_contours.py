import cv2
import numpy as np
from typing import List
from grade_contur import ContourEvaluationResult, evaluate_contour

# Set of static contours that should be avoided

static_contours_candidates = {}
static_contours = {}
# if a contour is in the same position for 4 frames in a row, it is considered static and should be added to static_contours

previous_positions = []

def is_centers_equal(center1, center2):
    """ Return true if centers are really close to each other """
    return np.linalg.norm(np.array(center1) - np.array(center2)) < 2

# Function to find the top N candidates based on contour evaluation scores
def find_top_candidates(
    contours: List[np.ndarray],
    top_n: int = 3,
) -> List[ContourEvaluationResult]:
    """
    Find the top N candidates based on contour evaluation scores (sorted in descending order).

    Args:
        contours (List[np.ndarray]): List of contours to evaluate.
        prev_position (Optional[Tuple[int, int]]): The previous position of the object.
        top_n (int): The number of top candidates to return.

    Returns:
        List[ContourEvaluationResult]: The top N evaluated contours, sorted by score.
    """
    candidates = []

    for contour in contours:
        result = evaluate_contour(contour, previous_positions)
        candidates.append(result)
        # if there is a previous position, check if the current position is the same as the previous one
        if previous_positions and is_centers_equal(result.center, previous_positions[-1]):
            # It makes sense to not consider a contour static if it is in the same position as the previous one, because
            # it might be a moving object that is just moving very slowly, e.g the ball jumps on the spot or something
            continue
        # if the contour is in the same position for 4 frames in a row, it is considered static

        if result.center in static_contours_candidates:
            static_contours_candidates[result.center] += 1
            if static_contours_candidates[result.center] >= 4:
                static_contours[result.center] = 1
        else:
            static_contours_candidates[result.center] = 1


    # Sort candidates by score in descending order
    candidates.sort(key=lambda x: x.score, reverse=True)
    
    top_n_candidates = candidates[:top_n]
    
    # if any top candidate is close to a known static contour, decrease it's score significantly
    # use the distance between the centers of the contours for comparison
    for candidate in top_n_candidates:
        for static_center in static_contours_candidates:
            # distance between the centers of the contours
            candidate_center = candidate.center
            if is_centers_equal(candidate_center, static_center):
                print(f"Static contour found at {static_center}, candidate at {candidate_center}, punishing candidate")
                candidate.score *= 0.001

    # resort the candidates after the punishment
    candidates.sort(key=lambda x: x.score, reverse=True)
    top_n_candidates = candidates[:top_n]
    
    previous_positions.append(candidates[0].center)

    return top_n_candidates


def get_contours(grayscale_image: np.ndarray, origin_path: str) -> np.ndarray:
    """
    Detect contours in a grayscale image and return the output image with contours drawn on it.
    """
    # Detect contours in the mask
    contours, _ = cv2.findContours(
        grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # # Draw the contours on the original image
    # output_image = cv2.imread(origin_path)
    # cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

    # # Mark a red circle around the detected contours
    # # print(f'Number of contours detected: {len(contours)}')
    # for contour in contours:
    #     (x, y), radius = cv2.minEnclosingCircle(contour)
    #     center = (int(x), int(y))
    #     radius = int(radius)
    #     cv2.circle(output_image, center, radius, (0, 0, 255), 2)


    return contours
