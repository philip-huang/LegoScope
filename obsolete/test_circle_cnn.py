import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from train_circle_cnn import CircleCNN  # Ensure this is the correct import for your model
import os
import re

# Load the trained model
model_path = "circle_detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CircleCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # Set model to evaluation mode

# Image folder and labels file
image_folder = "data/set-5"
labels_file = "data/set-5/labels.txt"

# Load labels
# labels = np.loadtxt(labels_file, delimiter=',')

# Get valid image filenames (only numbered .png files)
image_filenames = sorted(
    [f for f in os.listdir(image_folder) if re.fullmatch(r'\d+\.png', f)],
    key=lambda x: int(x.split('.')[0])
)

# Loop through each image
for img_idx, img_filename in enumerate(image_filenames):
    img_path = os.path.join(image_folder, img_filename)

    # Load and preprocess image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (480, 480))  # Resize to match training size
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 480, 480) 
    image_tensor = image_tensor / 255.0  # Normalize
    image_tensor = image_tensor.to(device)

    # Run inference
    with torch.no_grad():
        predicted_center = model(image_tensor).cpu().numpy()[0]  # Get (x, y) prediction

    # Convert predicted coordinates to integer
    pred_x, pred_y = int(predicted_center[0]), int(predicted_center[1])

    # Draw the predicted center on the image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for colored dot
    cv2.circle(image_color, (pred_x, pred_y), 5, (0, 0, 255), -1)  # Red dot

    # Show the image with prediction
    cv2.imshow("img", image_color)
    
    # Wait for key press, close on 'q'
    key = cv2.waitKey(0)
    if key == ord('q'):  # Press 'q' to exit early
        break

cv2.destroyAllWindows()
