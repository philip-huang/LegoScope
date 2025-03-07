#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
import time
from collections import deque
import rospy
from std_msgs.msg import Float32MultiArray
from scipy.optimize import least_squares
from find_cam import find_cam
from detect_offset import compute_offset, process_results
import detect_light_ring
import os
from datetime import datetime
last_mos = np.array([0,0,0])
#last_mos_queue = deque(maxlen=10)


CHECK_TILT = True
SAVE_CLIP = False
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

    filtered_center = np.array([0,0])
    print(37)
    if SAVE_CLIP:
        save_dir = os.path.join("saved_clips", "clip" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    print(43)
    rospy.init_node('tool_offset_publisher', anonymous=True)
    tool_offset_pub = rospy.Publisher('tool_offset', Float32MultiArray, queue_size=2)
    block_tilt_pub = rospy.Publisher('block_tilt', Float32MultiArray, queue_size=2)
    # Set the loop rate (e.g., 10 Hz)
    rate = rospy.Rate(8)
    print(49)
    while not rospy.is_shutdown():
        # Get the detected offset
        try:
            print(44)
            offset = compute_offset(camera, model, save_visual = save_dir )
        except:
            offset = None
        print("offset is: ", offset)
        if CHECK_TILT:
            tilt = detect_light_ring.detect_lightring(camera, model=light_ring_model, z = 30, e_center=[210,270])
        print(offset)
        # Publish only if offset is not None
        # If offset is None (no feature detected), controller's offset isn't updated
        if offset is not None:
            # Create a Float32MultiArray message
            msg = Float32MultiArray()
            msg.data = offset
            #print("published")
            # Publish the message
            tool_offset_pub.publish(msg)
        if CHECK_TILT:
            if tilt is not None:
                msg = Float32MultiArray()
                msg.data = tilt
                print("published")
        # Sleep to maintain the loop rate
        rate.sleep() 
