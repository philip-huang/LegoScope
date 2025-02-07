import os
import cv2
import numpy as np

# Define paths
data_folder = "data/set-1"
output_folder = "data/set-1/temp"
center_file = os.path.join(data_folder, "labels.txt")

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Read the center coordinates
with open(center_file, "r") as f:
    centers = [list(map(int, line.strip().split(","))) for line in f.readlines()]

# Process each image
for i, (x, y) in enumerate(centers):
    img_path = os.path.join(data_folder, f"{i}.png")
    output_path = os.path.join(output_folder, f"doted_{i}.png")

    # Load the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue  # Skip if image not found

    # Convert grayscale to BGR for red dot overlay
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw the red dot
    cv2.circle(img_bgr, (x, y), 3, (0, 0, 255), -1)

    # Save the modified image
    cv2.imwrite(output_path, img_bgr)

# Task completed
"Processing completed. Images saved in data/set-1/temp/"
