#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
from collections import deque
# from circle_fit import taubinSVD
from scipy.optimize import minimize
import detect_light_ring
from shapely.geometry import Polygon, Point
import os
import time
from datetime import datetime
last_mos = np.array([0,0,0])
last_mos_queue = deque(maxlen=5)
TOOL_CENTER = np.array([160, 265])
SAVE_CLIP = False
NEW_CAMERA = False
def mask_in_circle(mask_points, cx, cy, radius):
    """
    Compute the percentage of the mask area that lies inside a given circle.
    
    Parameters:
    - mask_points: List of (x, y) tuples representing the polygon vertices of the mask.
    - cx, cy: Center of the circle.
    - radius: Radius of the circle.
    
    Returns:
    - float: Percentage of the mask area inside the circle.
    """
    # Convert the mask into a polygon
    mask_polygon = Polygon(mask_points)
    
    # Define the circle as a shapely object
    circle = Point(cx, cy).buffer(radius)  # This creates a circular polygon
    
    # Compute the intersection of the mask with the circle
    intersection = mask_polygon.intersection(circle)
    
    # Compute areas
    mask_area = mask_polygon.area
    intersection_area = intersection.area
    
    # Compute percentage
    percentage_inside = (intersection_area / mask_area) * 1 if mask_area > 0 else 0
    
    return percentage_inside

def polygon_area(points):
    """
    Compute the area of a polygon given its vertices using the Shoelace theorem.
    
    Parameters:
    - points: List of (x, y) tuples representing the polygon vertices.
    
    Returns:
    - float: The area of the polygon.
    """
    points = np.array(points)  # Convert list to NumPy array for easier indexing
    x = points[:, 0]
    y = points[:, 1]
    
    # Apply Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    
    return area

def process_results2(results):
    transform = np.array([0.01, 0.01, 1])
    global last_mos
    if len(results) == 1:
        result = results[0]
        if not (results[0].keypoints.conf is None):
            mid_of_studs = results[0].keypoints.xy[0,1,:]
            mid_of_studs_conf = results[0].keypoints.conf[0,1]
            last_mos = 0.8 * last_mos + 0.2 * np.array(mid_of_studs.to('cpu'))
            output = np.zeros(3)
            output[:2] = last_mos
            output[2] = mid_of_studs_conf.cpu().item()
            #output = output @ transform.T
            print("center: ", last_mos, "with confidence", mid_of_studs_conf.cpu().item())
        else:
            print("no lego keypoints detected")
            output = np.zeros(3)
        
        return output

def cost_function(r, line_x, line_y, contour):
    """
    Compute the cost for a given radius r.
    """
    if NEW_CAMERA:
        expected_radius = 50
    else:
        expected_radius = 60
    radius = r[0]
    # Update cx and cy based on the given conditions
    cx = line_x - radius if contour[:, 0].mean() < line_x else line_x + radius
    cy = line_y - radius if contour[:, 1].mean() < line_y else line_y + radius

    # Compute distances and enclosed ratio
    distances = np.linalg.norm(contour - [cx, cy], axis=1)
    #outlier_ratio = np.mean(distances > radius)

    outlier_ratio = 1 - mask_in_circle(contour, cx, cy, radius)
    
    # Compute cost
    total_area = np.pi * radius ** 2  # Area (squared diameter)
    expected_area = np.pi * expected_radius ** 2
    area_diff = total_area - expected_area
    cost = 3e5 * outlier_ratio + area_diff
    return cost

def optimize_radius(line_x, line_y, contour, initial_guess):
    """
    Optimize the radius to minimize the cost function.
    """
    if NEW_CAMERA:
        result = minimize(
            cost_function, 
            x0=[initial_guess],  # Initial guess for radius
            args=(line_x, line_y, contour),  # Additional arguments for the cost function
            bounds=[(20,120)]  # Radius must be positive
        )
    else:
        result = minimize(
            cost_function, 
            x0=[initial_guess],  # Initial guess for radius
            args=(line_x, line_y, contour),  # Additional arguments for the cost function
            bounds=[(80,200)]  # Radius must be positive
        )
    optimized_radius = result.x[0]
    return optimized_radius, result.fun

def min_enclosing_circle_tangent_to_lines(contour, line_x, line_y):
    """
    Find the minimum enclosing circle of a contour that is tangent to both a vertical and a horizontal line.
    
    Parameters:
        contour (np.ndarray): Contour points as a numpy array of shape (n, 2).
        line_x (float): X-coordinate of the vertical line.
        line_y (float): Y-coordinate of the horizontal line.
    
    Returns:
        tuple: (center, radius) where center is (x, y) and radius is the circle radius.
    """
    # Find the initial minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    # Adjust the x-coordinate of the circle center for vertical line tangency
    if cx < line_x:
        cx = line_x -radius
    else:
        cx = line_x + radius

    # Adjust the y-coordinate of the circle center for horizontal line tangency
    if cy < line_y:
        cy = line_y -radius
    else:
        cy = line_y + radius
    #optimize
    radius, f = optimize_radius(line_x, line_y, contour, radius)
    if cx < line_x:
        cx = line_x -radius
    else:
        cx = line_x + radius

    # Adjust the y-coordinate of the circle center for horizontal line tangency
    if cy < line_y:
        cy = line_y -radius
    else:
        cy = line_y + radius
    # Return the adjusted circle center and radius
    return np.array((cx, cy)).astype(int), int(radius)
def min_enclosing_circle_tangent_to_lines_old(contour, line_x, line_y):
    """
    Find the minimum enclosing circle of a contour that is tangent to both a vertical and a horizontal line.
    
    Parameters:
        contour (np.ndarray): Contour points as a numpy array of shape (n, 2).
        line_x (float): X-coordinate of the vertical line.
        line_y (float): Y-coordinate of the horizontal line.
    
    Returns:
        tuple: (center, radius) where center is (x, y) and radius is the circle radius.
    """
    # Find the initial minimum enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    # Adjust the x-coordinate of the circle center for vertical line tangency
    if cx < line_x:
        cx = line_x -radius
    else:
        cx = line_x + radius

    # Adjust the y-coordinate of the circle center for horizontal line tangency
    if cy < line_y:
        cy = line_y -radius
    else:
        cy = line_y + radius

    # Recalculate the radius to ensure all points are enclosed
    distances = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    radius = np.max(distances)
    if cx < line_x:
        cx = line_x -radius
    else:
        cx = line_x + radius
    if cy < line_y:
        cy = line_y -radius
    else:
        cy = line_y + radius

    # Return the adjusted circle center and radius
    return np.array((cx, cy)).astype(int), int(radius)

def process_results(result, conf, angle, verbose=False):
    global last_mos_queue
    
    # Convert the new measurement to CPU and append to the queue
    if conf > 0.3:
        new_measurement = np.concatenate([result, [angle]])
        last_mos_queue.append(new_measurement)
        
        # Compute the rolling average
        rolling_average = np.median(np.array(last_mos_queue), axis=0)

        output = np.zeros(3)
        output[:3] = rolling_average

        if verbose:
            print("center: ", rolling_average, "with confidence", conf)
            
        return output
    else:
        return None
    
def compute_offset(camera, model, fx = 1100 , fy = 1100, z = 30.0, show_yolo = False, visualize = False, 
                   visualize_all = False,save_visual = None, crosshair = None, tool_center = TOOL_CENTER):
    '''
    Arguments: 
    camera to read from, and yolo keypoint model to use
    camera focal and depth in mm. Assume cx, cy are at image center, and assume fixed z
    Outputs:
    [x,y,confidence] if detected, None otherwise
    '''
    t0 = time.perf_counter()
    ret, og_frame = camera.read()
    
    # og_frame = cv2.imread("sample.jpg")
    if ret:
        h, w, c = og_frame.shape
    else:
        print("read frame file")
        return None
    start_w = (w - 480) // 2
    end_w = start_w + 480
    og_frame = og_frame[:, start_w:end_w, :]
    if crosshair:
        center_x, center_y = map(int, crosshair)
        crosshair_length = 200  # Length of the crosshair lines
        color = (0,0,0)  # Green color
        thickness = 2  # Line thickness
        cv2.line(og_frame, (center_x, center_y - crosshair_length), 
                (center_x, center_y + crosshair_length), color, thickness)
        cv2.line(og_frame, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), color, thickness)
        cv2.imshow("manual mode frame", og_frame)
        cv2.waitKey(1)
        return
    
    t1 = time.perf_counter()
    results = model.predict(og_frame, show = show_yolo, conf = 0.4, iou = 0.3, verbose = False)
    
    t2 = time.perf_counter()
    if results[0].masks is None:
        print("No Studs detected")
        if save_visual is not None:
            save_dir = save_visual
            frame_count = len(os.listdir(save_dir)) + 1
            save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
            cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
            cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
            # cv2.imshow("og_frame", og_frame)
            cv2.imwrite(save_path, og_frame)
        return None
    mask = results[0].masks.xy
    conf = torch.min(results[0].boxes.conf).to('cpu').item()
    if (len(results[0].masks) < 2):
        if save_visual is not None:
            save_dir = save_visual
            frame_count = len(os.listdir(save_dir)) + 1
            save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
            cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
            cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
            # cv2.imshow("og_frame", og_frame)
            cv2.imwrite(save_path, og_frame)
        print(len(results[0].masks), " studs detected, abort computing offset")
        return None
    elif (len(results[0].masks) > 2):   
        mask_dists= []#todo: confidence + distance co
        for i in range(len(mask)):
            mask_center = np.median(mask[i], axis = 0)
            if polygon_area(mask[i]) > 6500:
                
                if NEW_CAMERA:
                    dx = mask_center[0] - tool_center[0]
                    dy = mask_center[0] - tool_center[1]
                    distance = np.sqrt(3 * dx**2 + dy**2)
                else:
                    distance = np.linalg.norm(mask_center - tool_center)
                mask_dists.append(distance)
        
        mask_indices = np.argsort(mask_dists)
        print(mask_dists)
        if len(mask_indices) < 2:
            if save_visual is not None:
                save_dir = save_visual
                frame_count = len(os.listdir(save_dir)) + 1
                save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
                cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
                cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
                # cv2.imshow("og_frame", og_frame)
                cv2.imwrite(save_path, og_frame)
            return None
        mask = [mask[mask_indices[0]], mask[mask_indices[1]]]
        #mask = mask[0:2]
        print("warning: ",len(results[0].masks), " studs detected")
    segments = np.zeros(og_frame.shape[:2])
    centers = []
    radiuses = []
    for i in range(min(2, len(mask))):
        (x, y), radius = cv2.minEnclosingCircle(mask[i])
        center = (int(x), int(y))  # Convert center coordinates to integers
        radius = int(radius)       # Convert radius to an integer
        centers.append(center)
        radiuses.append(radius)
    top_stud_mask = mask[np.argmax([np.median(mask[0][:,1]), np.median(mask[1][:,1])])] #top stud is the one with higher Y value, which is at the BOTTOM in image
    bottom_stud_mask = mask[np.argmin([np.median(mask[0][:,1]), np.median(mask[1][:,1])])]
    y_top = np.min(top_stud_mask[:,1]).astype(int)
    y_bottom = np.max(bottom_stud_mask[:,1]).astype(int)
    if(np.average([centers[0][0], centers[1][0]])< 200):
        top_stud_side_x = np.max(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.max(bottom_stud_mask[:,0]).astype(int)
    else:
        top_stud_side_x = np.min(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.min(bottom_stud_mask[:,0]).astype(int)

    top_stud_center_adj, top_stud_radius_adj = min_enclosing_circle_tangent_to_lines(top_stud_mask, top_stud_side_x, y_top)
    bottom_stud_center_adj, bottom_stud_radius_adj = min_enclosing_circle_tangent_to_lines(bottom_stud_mask, bottom_stud_side_x, y_bottom)

    target_center_adj = np.mean([top_stud_center_adj, bottom_stud_center_adj], axis = 0)
    t3 = time.perf_counter()

    studs_diff = top_stud_center_adj - bottom_stud_center_adj
    angle = np.arctan2(studs_diff[1], studs_diff[0])
    angle -= 1.5343
    output = process_results(target_center_adj, conf,angle)
    if output is None:
        print("low confidence, ignoring offset")
        return None
    output[:2] -= tool_center#diff from center
    output[:2] *= z
    output[:2] /= np.array([fx, fy])

    #temporary scaling of offset for tuning: x,y scale should be -1 and yaw should be 1
    output[0] *= -1 #x,y is reversed
    output[1] *= -1 
    output[2] *= 1
    if save_visual is not None:
        #print(441)
        save_dir = save_visual
        #print(444)
        frame_count = len(os.listdir(save_dir)) + 1
        save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
        cv2.circle(og_frame, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, target_center_adj.astype(np.int64), 6, [0,230,0], -1)
        textx = f"x: {output[0]:.2f} mm"
        texty = f"y: {output[1]:.2f} mm"
        textth = f"yaw: {output[2]:.2f} rad"
        (text_w, text_h), _ = cv2.getTextSize(textth, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        box_x, box_y = og_frame.shape[1] - text_w - 20, 20
        cv2.line(og_frame, (tool_center[0], 0), (tool_center[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, tool_center[1]), (og_frame.shape[1], tool_center[1]), (0, 0, 0), 2)
        cv2.rectangle(og_frame, (box_x - 10, box_y - 10), (box_x + text_w + 5, box_y + text_h * 3 + 13), (0, 0, 0), -1)
        cv2.putText(og_frame, textx, (box_x, box_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, texty, (box_x, box_y + 2 * text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, textth, (box_x, box_y + 3 * text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        # cv2.imshow("og_frame", og_frame)
        cv2.imwrite(save_path, og_frame)
    if visualize:
        
        if visualize_all:
            cv2.imshow("og_frame", og_frame)

            segments = cv2.cvtColor(segments.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            segments = cv2.fillPoly(segments, [top_stud_mask.astype(np.int32)], (200, 200, 200))
            segments = cv2.fillPoly(segments, [bottom_stud_mask.astype(np.int32)], (200, 200, 200))
            cv2.imshow("og_masks", segments)
            
            cv2.line(segments, (top_stud_side_x, y_top), (top_stud_side_x, segments.shape[0]), (0, 0, 200), 2)
            cv2.line(segments, (bottom_stud_side_x, 0), (bottom_stud_side_x, y_bottom), (0, 0, 200), 2)
            cv2.line(segments, (0, y_top), (segments.shape[1], y_top), (0, 0, 200), 2)
            cv2.line(segments, (0, y_bottom), (segments.shape[1], y_bottom), (0, 0, 200), 2)
            cv2.imshow("masks_with_lines", segments)

            cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
            cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
            cv2.circle(segments, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
            cv2.circle(segments, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
            cv2.imshow("masks_with_circles", segments)

        cv2.circle(og_frame, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, target_center_adj.astype(np.int64), 6, [0,230,0], -1)
        
        
        textx = f"x: {output[0]:.2f} mm"
        texty = f"y: {output[1]:.2f} mm"
        textth = f"yaw: {output[2]:.2f} rad"
        (text_w, text_h), _ = cv2.getTextSize(textth, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        box_x, box_y = og_frame.shape[1] - text_w - 20, 20
        cv2.line(og_frame, (tool_center[0], 0), (tool_center[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, tool_center[1]), (og_frame.shape[1], tool_center[1]), (0, 0, 0), 2)
        cv2.rectangle(og_frame, (box_x - 10, box_y - 10), (box_x + text_w + 5, box_y + text_h * 3 + 13), (0, 0, 0), -1)
        cv2.putText(og_frame, textx, (box_x, box_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, texty, (box_x, box_y + 2 * text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, textth, (box_x, box_y + 3 * text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow("og_frame", og_frame)
        cv2.waitKey(20)


    return np.concatenate([output[:2] / 1000, [output[2]]]) #mm to meter

def compute_offset_image(og_frame, model, fx = 1100 , fy = 1100, z = 30.0, show_yolo = False, visualize = False, 
                         visualize_all = False,save_visual = None, crosshair = None, ros_pub = None, tool_center = TOOL_CENTER):
    '''
    Arguments: 
    camera to read from, and yolo keypoint model to use
    camera focal and depth in mm. Assume cx, cy are at image center, and assume fixed z
    Outputs:
    [x,y,confidence] if detected, None otherwise
    '''
    h, w, c = og_frame.shape
    start_w = (w - 480) // 2
    end_w = start_w + 480
    og_frame = og_frame[:, start_w:end_w, :]
    if crosshair:
        center_x, center_y = map(int, crosshair)
        crosshair_length = 200  # Length of the crosshair lines
        color = (0,0,0)  # Green color
        thickness = 2  # Line thickness
        cv2.line(og_frame, (center_x, center_y - crosshair_length), 
                (center_x, center_y + crosshair_length), color, thickness)
        cv2.line(og_frame, (center_x - crosshair_length, center_y), (center_x + crosshair_length, center_y), color, thickness)
        cv2.imshow("manual mode frame", og_frame)
        cv2.waitKey(1)
        return
    
    t1 = time.perf_counter()
    results = model.predict(og_frame, show = show_yolo, conf = 0.4, iou = 0.3, verbose = False)
    
    t2 = time.perf_counter()
    if results[0].masks is None:
        print("No Studs detected")
        if save_visual is not None:
            save_dir = save_visual
            frame_count = len(os.listdir(save_dir)) + 1
            save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
            cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
            cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
            # cv2.imshow("og_frame", og_frame)
            cv2.imwrite(save_path, og_frame)
        return None
    mask = results[0].masks.xy
    conf = torch.min(results[0].boxes.conf).to('cpu').item()
    if (len(results[0].masks) < 2):
        if save_visual is not None:
            save_dir = save_visual
            frame_count = len(os.listdir(save_dir)) + 1
            save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
            cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
            cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
            # cv2.imshow("og_frame", og_frame)
            cv2.imwrite(save_path, og_frame)
        print(len(results[0].masks), " studs detected, abort computing offset")
        return None
    elif (len(results[0].masks) > 2):   
        mask_dists= []#todo: confidence + distance co
        for i in range(len(mask)):
            mask_center = np.median(mask[i], axis = 0)
            if polygon_area(mask[i]) > 6500:
                
                if NEW_CAMERA:
                    dx = mask_center[0] - tool_center[0]
                    dy = mask_center[0] - tool_center[1]
                    distance = np.sqrt(3 * dx**2 + dy**2)
                else:
                    distance = np.linalg.norm(mask_center - tool_center)
                mask_dists.append(distance)
        
        mask_indices = np.argsort(mask_dists)
        print(mask_dists)
        if len(mask_indices) < 2:
            if save_visual is not None:
                save_dir = save_visual
                frame_count = len(os.listdir(save_dir)) + 1
                save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
                cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
                cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
                # cv2.imshow("og_frame", og_frame)
                cv2.imwrite(save_path, og_frame)
            return None
        mask = [mask[mask_indices[0]], mask[mask_indices[1]]]
        #mask = mask[0:2]
        print("warning: ",len(results[0].masks), " studs detected")
    segments = np.zeros(og_frame.shape[:2])
    centers = []
    radiuses = []
    for i in range(min(2, len(mask))):
        (x, y), radius = cv2.minEnclosingCircle(mask[i])
        center = (int(x), int(y))  # Convert center coordinates to integers
        radius = int(radius)       # Convert radius to an integer
        centers.append(center)
        radiuses.append(radius)
    top_stud_mask = mask[np.argmax([np.median(mask[0][:,1]), np.median(mask[1][:,1])])] #top stud is the one with higher Y value, which is at the BOTTOM in image
    bottom_stud_mask = mask[np.argmin([np.median(mask[0][:,1]), np.median(mask[1][:,1])])]
    y_top = np.min(top_stud_mask[:,1]).astype(int)
    y_bottom = np.max(bottom_stud_mask[:,1]).astype(int)
    if(np.average([centers[0][0], centers[1][0]])< 200):
        top_stud_side_x = np.max(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.max(bottom_stud_mask[:,0]).astype(int)
    else:
        top_stud_side_x = np.min(top_stud_mask[:,0]).astype(int)
        bottom_stud_side_x = np.min(bottom_stud_mask[:,0]).astype(int)

    top_stud_center_adj, top_stud_radius_adj = min_enclosing_circle_tangent_to_lines(top_stud_mask, top_stud_side_x, y_top)
    bottom_stud_center_adj, bottom_stud_radius_adj = min_enclosing_circle_tangent_to_lines(bottom_stud_mask, bottom_stud_side_x, y_bottom)

    target_center_adj = np.mean([top_stud_center_adj, bottom_stud_center_adj], axis = 0)
    t3 = time.perf_counter()

    studs_diff = top_stud_center_adj - bottom_stud_center_adj
    angle = np.arctan2(studs_diff[1], studs_diff[0])
    angle -= 1.5343
    output = process_results(target_center_adj, conf,angle)
    if output is None:
        print("low confidence, ignoring offset")
        return None
    output[:2] -= tool_center#diff from center
    output[:2] *= z
    output[:2] /= np.array([fx, fy])

    #temporary scaling of offset for tuning: x,y scale should be -1 and yaw should be 1
    output[0] *= -1 #x,y is reversed
    output[1] *= -1 
    output[2] *= 1
    if save_visual is not None:
        #print(441)
        save_dir = save_visual
        #print(444)
        frame_count = len(os.listdir(save_dir)) + 1
        save_path = os.path.join(save_dir, f"frame{frame_count:04d}.png")
        cv2.circle(og_frame, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, target_center_adj.astype(np.int64), 6, [0,230,0], -1)
        textx = f"x: {output[0]:.2f} mm"
        texty = f"y: {output[1]:.2f} mm"
        textth = f"yaw: {output[2]:.2f} rad"
        (text_w, text_h), _ = cv2.getTextSize(textth, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        box_x, box_y = og_frame.shape[1] - text_w - 20, 20
        cv2.line(og_frame, (tool_center[0], 0), (tool_center[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, tool_center[1]), (og_frame.shape[1], tool_center[1]), (0, 0, 0), 2)
        cv2.rectangle(og_frame, (box_x - 10, box_y - 10), (box_x + text_w + 5, box_y + text_h * 3 + 13), (0, 0, 0), -1)
        cv2.putText(og_frame, textx, (box_x, box_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, texty, (box_x, box_y + 2 * text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, textth, (box_x, box_y + 3 * text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        # cv2.imshow("og_frame", og_frame)
        cv2.imwrite(save_path, og_frame)
    if visualize:
        
        if visualize_all:
            cv2.imshow("og_frame", og_frame)

            segments = cv2.cvtColor(segments.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            segments = cv2.fillPoly(segments, [top_stud_mask.astype(np.int32)], (200, 200, 200))
            segments = cv2.fillPoly(segments, [bottom_stud_mask.astype(np.int32)], (200, 200, 200))
            cv2.imshow("og_masks", segments)
            
            cv2.line(segments, (top_stud_side_x, y_top), (top_stud_side_x, segments.shape[0]), (0, 0, 200), 2)
            cv2.line(segments, (bottom_stud_side_x, 0), (bottom_stud_side_x, y_bottom), (0, 0, 200), 2)
            cv2.line(segments, (0, y_top), (segments.shape[1], y_top), (0, 0, 200), 2)
            cv2.line(segments, (0, y_bottom), (segments.shape[1], y_bottom), (0, 0, 200), 2)
            cv2.imshow("masks_with_lines", segments)

            cv2.line(og_frame, (og_frame.shape[1] // 2, 0), (og_frame.shape[1] // 2, og_frame.shape[0]), (0, 0, 0), 2)
            cv2.line(og_frame, (0, og_frame.shape[0] // 2), (og_frame.shape[1], og_frame.shape[0] // 2), (0, 0, 0), 2)
            cv2.circle(segments, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
            cv2.circle(segments, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
            cv2.imshow("masks_with_circles", segments)

        cv2.circle(og_frame, top_stud_center_adj,top_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, bottom_stud_center_adj,bottom_stud_radius_adj, [0,230,0], 2)
        cv2.circle(og_frame, target_center_adj.astype(np.int64), 6, [0,230,0], -1)
        
        
        textx = f"x: {output[0]:.2f} mm"
        texty = f"y: {output[1]:.2f} mm"
        textth = f"yaw: {output[2]:.2f} rad"
        (text_w, text_h), _ = cv2.getTextSize(textth, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        box_x, box_y = og_frame.shape[1] - text_w - 20, 20
        cv2.line(og_frame, (tool_center[0], 0), (tool_center[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, tool_center[1]), (og_frame.shape[1], tool_center[1]), (0, 0, 0), 2)
        cv2.rectangle(og_frame, (box_x - 10, box_y - 10), (box_x + text_w + 5, box_y + text_h * 3 + 13), (0, 0, 0), -1)
        cv2.putText(og_frame, textx, (box_x, box_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, texty, (box_x, box_y + 2 * text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, textth, (box_x, box_y + 3 * text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        #cv2.imshow("og_frame", og_frame)
        #cv2.waitKey(20)

        if ros_pub is not None:
            from cv_bridge import CvBridge
            bridge = CvBridge()
            msg = bridge.cv2_to_imgmsg(og_frame, encoding="bgr8")
            ros_pub.publish(msg)


    return np.concatenate([output[:2] / 1000, [output[2]]]) #mm to meter

if __name__ == "__main__":
    model = YOLO("models/studs-seg2.pt")
    light_ring_model = YOLO("models/lightringv2.pt")
    camera = cv2.VideoCapture(4)
    filtered_center = np.array([0,0])
    if SAVE_CLIP:
        save_dir = os.path.join("saved_clips", "clip" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    while True:
        # Get the detected offset
        offset = compute_offset(camera, model, show_yolo=True, visualize=True, visualize_all= False, save_visual= save_dir)
        #detect_light_ring.detect_lightring(camera, light_ring_model, 30, e_center= [210, 270], visualize= True)
        print(offset)
