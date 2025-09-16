#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
from collections import deque

last_angle = np.array([0,0])
last_angle_queue = deque(maxlen=6)

def find_center(xyxy, img_center):
    
    h_lim = [30,300]
    w_lim = [30,300]
    ratio_lim = 1.8
    center_tolerance = 100

    x1, y1, x2, y2 = map(int, xyxy)
    h, w = (x2-x1, y2-y1)
    cx, cy = ((x1 + x2)/2, (y1 + y2)/2)
    #print("cycy: ", cx, " ", cy)
    if h < h_lim[0] or h > h_lim[1]:
         return None
    if w < w_lim[0] or w > w_lim[1]:
         return None
    if w/h > ratio_lim or h/w > ratio_lim:
         return None
    return np.array([cx - img_center[0], cy - img_center[1]])
def check_centered(cxcy):
    offset = np.array([0,0])
    tol = np.array([10,10])
    xy = cxcy + offset
    z = 30
    fy = 1000
    xtilt = xy[0] / 20
    ytilt = np.rad2deg(np.arctan(xy[1] / fy))
    #print("X tilt: ", xtilt)
    #print("Y tilt: ", ytilt)
    return [xtilt, ytilt]
    
def detect_lightring(camera, model, z, e_center = [240,240],  visualize = False):
    global last_angle_queue
    ret, og_frame = camera.read()
    h, w, c = og_frame.shape

# Calculate the start and end points for the width dimension to get the middle 480 pixels
    start_w = (w - 480) // 2
    end_w = start_w + 480

    # Crop the middle section
    og_frame = og_frame[:, start_w:end_w, :]
    results = model.predict(og_frame, show = False, conf = 0.6, verbose = False, device = 'cuda')
    tilt = None
    if len(results[0].boxes) > 0:
        # light_ring = results[0].boxes[0]
        # box = light_ring.xyxy[0]  # x1, y1, x2, y2 format
        # x1, y1, x2, y2 = map(int, box)
        # confidence = light_ring.conf.item()
        # cv2.rectangle(og_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        #                 # Put confidence text
        # cv2.putText(og_frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for detection in results[0].boxes:  # Accessing the detections
            confidence = detection.conf.item()
            if confidence > 0.3:
                box = detection.xyxy[0]  # x1, y1, x2, y2 format
                x1, y1, x2, y2 = map(int, box)  # Convert to integers for OpenCV
                # Draw the rectangle on the original frame
                cv2.rectangle(og_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                # Put confidence text
                cv2.putText(og_frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                center = find_center(detection.xyxy[0], e_center)
                
                if center is not None:
                    #print(center)
                    tilt = check_centered(center)
                    last_angle_queue.append(tilt)
    rolling_average = np.median(np.array(last_angle_queue), axis=0)                 
    if visualize:
        
        
        cv2.line(og_frame, (e_center[0], 0), (e_center[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, e_center[1]), (og_frame.shape[1], e_center[1]), (0, 0, 0), 2)
        cv2.circle(og_frame, (center + e_center).astype(np.int64), 8, [0,230,0], -1)
        textp = f"pitch: {rolling_average[0]:.2f} deg"
        textr = f"roll: {rolling_average[1]:.2f} deg"
        (text_w, text_h), _ = cv2.getTextSize(textp, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        box_x, box_y = og_frame.shape[1] - text_w - 20, 20
        cv2.rectangle(og_frame, (box_x - 10, box_y - 10), (box_x + text_w + 5, box_y + text_h * 2 + 13), (0, 0, 0), -1)
        cv2.putText(og_frame, textp, (box_x, box_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, textr, (box_x, box_y + 2 * text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow("light_ring", og_frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            pass
    
    return rolling_average
              
     


# model = YOLO("yolov8n.pt")
if __name__ == "__main__":
    model = YOLO("lightringv2.pt")
    camera = cv2.VideoCapture(4)
    while True:
        print(detect_lightring(camera, model, 30, e_center= [222, 270], visualize= True))
# while True:
    
#     ret, og_frame = camera.read()
#     h, w, c = og_frame.shape

# # Calculate the start and end points for the width dimension to get the middle 480 pixels
#     start_w = (w - 480) // 2
#     end_w = start_w + 480

#     # Crop the middle section
#     og_frame = og_frame[:, start_w:end_w, :]
#     results = model.predict(og_frame, show = False, verbose = False, device = 'cpu')
#     if len(results[0].boxes) > 0:
#         # light_ring = results[0].boxes[0]
#         # box = light_ring.xyxy[0]  # x1, y1, x2, y2 format
#         # x1, y1, x2, y2 = map(int, box)
#         # confidence = light_ring.conf.item()
#         # cv2.rectangle(og_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
#         #                 # Put confidence text
#         # cv2.putText(og_frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         for detection in results[0].boxes:  # Accessing the detections
#             confidence = detection.conf.item()
#             if confidence > 0.3:
#                 box = detection.xyxy[0]  # x1, y1, x2, y2 format
#                 x1, y1, x2, y2 = map(int, box)  # Convert to integers for OpenCV
#                 # Draw the rectangle on the original frame
#                 cv2.rectangle(og_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
#                 # Put confidence text
#                 cv2.putText(og_frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 center = find_center(detection.xyxy[0])
                
#                 if center is not None:
#                     print(center)
#                     check_centered(center)
                    
#     # t0 = time.perf_counter()
    
#     # print("infer time:", time.perf_counter() - t0)
#     cv2.imshow("Detections", og_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#             break