#use conda::base
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import numpy as np
import torch
from collections import deque
from scipy.optimize import least_squares
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
def fit_circle2(points):
    """
    Real-time Accurate Circle Fitting (RACF) algorithm implementation.

    Parameters:
        points (ndarray): Array of shape (N, 2) containing the x, y coordinates of the points.

    Returns:
        (float, float, float): Estimated circle parameters (cx, cy, r) where
        (cx, cy) is the center and r is the radius.
    """
    # Ensure points are in numpy array
    points = np.asarray(points)

    # Calculate the mean of the points
    x_mean = np.mean(points[:, 0])
    y_mean = np.mean(points[:, 1])

    # Center the points by subtracting the mean
    centered_points = points - np.array([x_mean, y_mean])

    # Formulate the matrices for least-squares fitting
    Z = np.hstack((
        centered_points[:, 0:1] ** 2 + centered_points[:, 1:2] ** 2,
        centered_points[:, 0:1],
        centered_points[:, 1:2],
        np.ones((points.shape[0], 1))
    ))

    # Solve the linear system Z.T * Z * a = 0 using SVD or least-squares
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    a = Vt.T[:, -1]  # Last column of Vt gives the solution

    # Recover circle parameters
    c = -0.5 * a[1:3] / a[0]
    radius = np.sqrt((c[0] ** 2 + c[1] ** 2) - (a[3] / a[0]))

    # Shift back the center to original coordinates
    c += np.array([x_mean, y_mean])

    return np.array(c).astype(int), int(radius)
def fit_circle(points):
    # Define the function to minimize
    def residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    # Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Initial guess (mean of points for center, average distance for radius)
    x_m, y_m = np.mean(x), np.mean(y)
    r_guess = np.mean(np.sqrt((x - x_m)**2 + (y - y_m)**2))
    initial_guess = [x_m, y_m, r_guess]

    # Least squares optimization
    result = least_squares(residuals, initial_guess, args=(x, y))
    xc, yc, r = result.x
    return np.array([xc, yc]).astype(int), int(r)
def cost_function(r, line_x, line_y, contour):
    """
    Compute the cost for a given radius r.
    """
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
    expected_area = np.pi * 60 ** 2
    area_diff = total_area - expected_area
    cost = 3e5 * outlier_ratio + area_diff
    return cost
def optimize_radius(line_x, line_y, contour, initial_guess):
    """
    Optimize the radius to minimize the cost function.
    """
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
    # Recalculate the radius to ensure all points are enclosed
    # distances = np.sqrt((contour[:, 0] - cx) ** 2 + (contour[:, 1] - cy) ** 2)
    # within_circle = distances < radius
    # enclosed_ratio = np.sum(within_circle) / within_circle.shape[0]
    # total_area = radius*2
    # cost = 10e5 * enclosed_ratio**2 + total_area
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

def find_intersections(centers, y_top, y_bottom):
    """
    Calculate the intersection points of Line1 with horizontal lines at y_top and y_bottom.
    
    Parameters:
        centers (list of tuples): Two points [(x0, y0), (x1, y1)] defining Line1.
        y_top (int): Y-coordinate of the horizontal line (Line2).
        y_bottom (int): Y-coordinate of the horizontal line (Line3).
    
    Returns:
        tuple: Intersection points ((x_intersect1, y_top), (x_intersect2, y_bottom)).
    """
    # Extract points for Line1
    (x0, y0), (x1, y1) = centers

    # Calculate slope (m1) and intercept (c1) of Line1
    if x1 != x0:  # Line1 is not vertical
        m1 = (y1 - y0) / (x1 - x0)
        c1 = y0 - m1 * x0
    else:  # Line1 is vertical
        m1 = float('inf')  # Infinite slope
        c1 = x0  # Vertical line's x-coordinate

    # Intersection with Line2 (y = y_top)
    if m1 != float('inf'):
        x_intersect1 = (y_top - c1) / m1
    else:
        x_intersect1 = c1  # For vertical Line1
    intersection1 = (int(round(x_intersect1)), y_top)

    # Intersection with Line3 (y = y_bottom)
    if m1 != float('inf'):
        x_intersect2 = (y_bottom - c1) / m1
    else:
        x_intersect2 = c1  # For vertical Line1
    intersection2 = (int(round(x_intersect2)), y_bottom)

    return np.asarray([intersection1, intersection2])

def process_results(result, conf, angle):
    transform = np.array([0.01, 0.01, 1])
    global last_mos_queue
    
    # Convert the new measurement to CPU and append to the queue
    if conf > 0.3:
        new_measurement = np.concatenate([result, [angle]])
        last_mos_queue.append(new_measurement)
        
        # Compute the rolling average
        rolling_average = np.median(np.array(last_mos_queue), axis=0)

        output = np.zeros(3)
        output[:3] = rolling_average

        print("center: ", rolling_average, "with confidence", conf)
            
        return output
    else:
        return None
    
def compute_offset(camera, model, fx = 1100 , fy = 1100, z = 30.0, show_yolo = False, visualize = False, visualize_all = False,save_visual = None, crosshair = None):
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
    # Crop the middle section
    # og_frame = cv2.resize(og_frame, [640,480])
    
    t1 = time.perf_counter()
    results = model.predict(og_frame, show = show_yolo, conf = 0.5, iou = 0.3, verbose = False)
    
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
            if polygon_area(mask[i]) > 8000:
                mask_dists.append(np.linalg.norm(mask_center - TOOL_CENTER))
        
        mask_indices = np.argsort(mask_dists)
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
    
    
    
        # cv2.waitKey(10)
        
    t3 = time.perf_counter()

    studs_diff = top_stud_center_adj - bottom_stud_center_adj
    angle = np.arctan2(studs_diff[1], studs_diff[0])
    angle -= 1.5343
    output = process_results(target_center_adj, conf,angle)
    if output is None:
        print("low confidence, ignoring offset")
        return None
    output[:2] -= TOOL_CENTER#diff from center
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
        cv2.line(og_frame, (TOOL_CENTER[0], 0), (TOOL_CENTER[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, TOOL_CENTER[1]), (og_frame.shape[1], TOOL_CENTER[1]), (0, 0, 0), 2)
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
        
        
        # cv2.circle(segments, top_stud_center_adj,top_stud_radius_adj, 250, 2)
        # cv2.circle(segments, bottom_stud_center_adj,top_stud_radius_adj, 250, 2)
        
        # Draw a black text box in the upper right corner
        textx = f"x: {output[0]:.2f} mm"
        texty = f"y: {output[1]:.2f} mm"
        textth = f"yaw: {output[2]:.2f} rad"
        (text_w, text_h), _ = cv2.getTextSize(textth, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
        box_x, box_y = og_frame.shape[1] - text_w - 20, 20
        cv2.line(og_frame, (TOOL_CENTER[0], 0), (TOOL_CENTER[0], og_frame.shape[0]), (0, 0, 0), 2)
        cv2.line(og_frame, (0, TOOL_CENTER[1]), (og_frame.shape[1], TOOL_CENTER[1]), (0, 0, 0), 2)
        cv2.rectangle(og_frame, (box_x - 10, box_y - 10), (box_x + text_w + 5, box_y + text_h * 3 + 13), (0, 0, 0), -1)
        cv2.putText(og_frame, textx, (box_x, box_y + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, texty, (box_x, box_y + 2 * text_h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(og_frame, textth, (box_x, box_y + 3 * text_h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow("og_frame", og_frame)
        cv2.waitKey(20)


    return np.concatenate([output[:2] / 1000, [output[2]]]) #mm to meter

if __name__ == "__main__":
    model = YOLO("studs-seg2.pt")
    light_ring_model = YOLO("lightringv2.pt")
    camera = cv2.VideoCapture(0)
    filtered_center = np.array([0,0])
    if SAVE_CLIP:
        save_dir = os.path.join("saved_clips", "clip" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
    # Set the loop rate (e.g., 10 Hz)
    while True:
        # Get the detected offset
        offset = compute_offset(camera, model, show_yolo=False, visualize=True, visualize_all= False, save_visual= save_dir)
        #detect_light_ring.detect_lightring(camera, light_ring_model, 30, e_center= [210, 270], visualize= True)
        print(offset)
