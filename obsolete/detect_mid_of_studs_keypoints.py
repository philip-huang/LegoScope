#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
last_dfc = np.array([0,0])
def find_center(xyxy):
    img_center = [240,240]
    h_lim = [30,300]
    w_lim = [30,300]
    ratio_lim = 1.8
    center_tolerance = 100

    x1, y1, x2, y2 = map(int, xyxy)
    h, w = (x2-x1, y2-y1)
    cx, cy = ((x1 + x2)/2, (y1 + y2)/2)

    if h < h_lim[0] or h > h_lim[1]:
         return None
    if w < w_lim[0] or w > w_lim[1]:
         return None
    if w/h > ratio_lim or h/w > ratio_lim:
         return None
    return np.array([cx - img_center[0], cy - img_center[1]])

def process_results(results):
    global last_dfc
    if len(results) == 1:
        result = results[0]
        if not (results[0].keypoints.conf is None):
            mid_of_studs = results[0].keypoints.xy[0,1,:]
            mid_of_studs_conf = results[0].keypoints.conf[0,1]
            diff_from_center = mid_of_studs - torch.tensor([208.5,276]).to(device=mid_of_studs.device)
            diff_from_center = diff_from_center.cpu().numpy()
            last_dfc = 0.8 * last_dfc + 0.2 * diff_from_center
            #print("mid_of_studs at ", mid_of_studs, "with confidence", mid_of_studs_conf)
            print("diff_from_center: ", last_dfc, "with confidence", mid_of_studs_conf.cpu().item())
            if np.linalg.norm(last_dfc) < 10:
                print("centered")
            elif np.linalg.norm(last_dfc) < 100:
                print("not centered")
            else:
                print("not attemtping pick up")
        else:
            print("no lego keypoints detected")
        output = np.zeros(3)
        output[:2] = last_dfc
        try:
            output[2] = mid_of_studs_conf.cpu().item()
        except:
             pass
        return output
        
# def check_centered(cxcy):
#     offset = np.array([36,-36])
#     tol = np.array([10,10])
#     xy = cxcy + offset
#     if abs(xy[0]) > tol[0]:
#         print("x tilt by ", xy[0])
#     if abs(xy[1]) > tol[1]:
#         print("y tilt by ", xy[1])
#     if abs(xy[0]) <= tol[0] and abs(xy[1]) <= tol[1]:
#         print("No Tilt")
    

    

              
     


# model = YOLO("yolov8n.pt")
model = YOLO("studs_keypoints.pt")
camera = cv2.VideoCapture(4)
filtered_center = np.array([0,0])

while True:
    
    ret, og_frame = camera.read()
    h, w, c = og_frame.shape

# Calculate the start and end points for the width dimension to get the middle 480 pixels
    start_w = (w - 480) // 2
    end_w = start_w + 480

    # Crop the middle section
    og_frame = og_frame[:, start_w:end_w, :]
    results = model.predict(og_frame, show = True, verbose = False, iou = 0.3)
    output = process_results(results)
        # light_ring = results[0].boxes[0]
        # box = light_ring.xyxy[0]  # x1, y1, x2, y2 format
        # x1, y1, x2, y2 = map(int, box)
        # confidence = light_ring.conf.item()
        # cv2.rectangle(og_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        #                 # Put confidence text
        # cv2.putText(og_frame, f'{confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
                    
    # t0 = time.perf_counter()
    
    # print("infer time:", time.perf_counter() - t0)
    cv2.imshow("Detections", og_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break