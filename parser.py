def parse_frame_data(text):
    frames = text.strip().split("--\n")
    result = []

    for i, frame in enumerate(frames):
        if not frame.strip():
            continue

        # Get frame number from the last line
        frame_info = frame.strip().split("\n")
        frame_number = int(frame_info[-1].split("frame")[-1].split(".")[0])

        # Parse the first contour (highest scoring) from the frame
        contour_line = frame_info[0]
        if "ContourEvaluationResult" in contour_line:
            # Extract center coordinates using string manipulation
            center_str = contour_line.split("center=(")[1].split(")")[0]
            x, y = map(int, center_str.split(", "))
            result.append((frame_number, x, y))

    return result


def write_coordinates_to_file(coordinates, filename="contour_centers.txt"):
    with open(filename, "w") as f:
        for frame_num, x, y in coordinates:
            f.write(f"frame{frame_num} ({x},{y})\n")


# Read from FramesCenter.txt
with open("FramesCenter.txt", "r") as f:
    data = f.read()

# Process the data and write to file
coordinates = parse_frame_data(data)
write_coordinates_to_file(coordinates)
