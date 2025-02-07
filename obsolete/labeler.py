import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
# import matplotlib
# matplotlib.use('Agg')
# Folder path
image_folder = "data/set-5"
center_file = os.path.join(image_folder, "labels.txt")
new_center_file = os.path.join(image_folder, "new_center.txt")

# Load centers
with open(center_file, "r") as f:
    centers = [list(map(int, line.strip().split(","))) for line in f]

# Get image filenames (assuming sorted numeric order)
image_files = [f"og{i}.png" for i in range(len(centers))]

# Store updated centers
new_centers = centers.copy()

# Image click callback function
def onclick(event):
    global new_centers, index, clicked
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        img_h, img_w = img.shape[:2]

        # Check if click is inside the image
        if 0 <= x < img_w and 0 <= y < img_h:
            new_centers[index] = [x, y]  # Update center
            clicked = True
        else:
            clicked = False  # No update

    plt.close()  # Close figure to move to next image

# Process each image
for index, img_name in enumerate(image_files):
    img_path = os.path.join(image_folder, img_name)
    
    if not os.path.exists(img_path):
        print(f"Skipping missing file: {img_name}")
        continue

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load grayscale image
    img_h, img_w = img.shape[:2]
    
    x, y = centers[index]
    
    fig, ax = plt.subplots(figsize=(11,11))
    ax.imshow(img, cmap="gray")
    ax.scatter([x], [y], color="red", s=30)  # Red dot
    
    clicked = False
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    # if index == 10:
    #     break

# Write new centers to new_center.txt
with open(new_center_file, "w") as f:
    for cx, cy in new_centers:
        f.write(f"{cx},{cy}\n")

print(f"Updated centers saved to {new_center_file}")
