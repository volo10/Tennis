import os
import re

import cv2
import procces_frame as pf
import numpy as np
from typing import List
from mp4_to_frames import extract_frames
from background_remover import remove_background
from get_countours_v2 import ContourTracker

ball_img_path = "images/raymond_ball.png"
ball_img_path_no_bg = "images/raymond_no_bg.png"
video_path = "videos/IMG_2916.mp4"
frames_dir = "raymond_frames"
proccess_frames_dir = f"raymond_proccessed_{frames_dir}"
output_path = "raymond_output.mp4"

dirs = [frames_dir, proccess_frames_dir]
# Create directories if they don't exist
for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)

# Function to sort filenames numerically
def sort_numerically(filenames):
    res = sorted(filenames, key=lambda x: int(re.search(r"\d+", x).group()))
    print(res)
    return res

def save_ball_game_with_no_backgrond():
    remove_background(ball_img_path, ball_img_path_no_bg)


def proccess_frames(frames_dir: str, tracker: ContourTracker) -> List[np.ndarray]:
    """
    Load a list of frames from the specified directory path.

    Args:
        frames_path (str): Path to the directory containing the frames.

    Returns:
        List[np.ndarray]: List of loaded frames.
    """
    frames = sort_numerically(os.listdir(frames_dir))
    frames_path = [os.path.join(frames_dir, frame) for frame in frames]
    for index, frame in enumerate(frames_path):
        proccess_frame_path = f'{proccess_frames_dir}/{frames[index]}'
        pf.proccess_frame(frame, ball_img_path_no_bg, proccess_frame_path,debug=True, tracker=tracker)

def create_video_from_frames(frames_dir: str, output_path: str) -> None:
    """
    Create a video from a list of frames.

    Args:
        frames_dir (str): Path to the directory containing the frames.
        output_path (str): Path to save the output video.
    """
    frames = sort_numerically(os.listdir(frames_dir))
    frames_path = [os.path.join(frames_dir, frame) for frame in frames if "filtered" not in frame]
    frame = cv2.imread(frames_path[0])
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for frame_path in frames_path:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()

# MAIN

if __name__ == "__main__":
    tracker = ContourTracker()
    extract_frames(video_path, frames_dir)
    save_ball_game_with_no_backgrond()
    proccess_frames(frames_dir, tracker)
    create_video_from_frames(proccess_frames_dir, output_path)