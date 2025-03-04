#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
import time
from collections import deque
from scipy.optimize import least_squares
from find_cam import find_cam
from detect_offset import compute_offset, process_results
import detect_light_ring
last_mos = np.array([0,0,0])
last_mos_queue = deque(maxlen=10)


CHECK_TILT = True
# TOOL_CENTER = np.array([170,265])

if __name__ == "__main__":
    model = YOLO("studs-seg2.pt")
    light_ring_model = YOLO("lightringv2.pt")
    camera_ids = find_cam()
    
    if camera_ids is None:
        print("no camera detected")
        assert False
    else:
        camera = cv2.VideoCapture(camera_ids[-1])
    camera = cv2.VideoCapture(1)
    filtered_center = np.array([0,0])


    while True:
        # Get the detected offset
        offset = compute_offset(camera, model, visualize=False, crosshair=[152,265])
        # Sleep to maintain the loop rate
