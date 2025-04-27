# Eye-in-Finger / LegoScope

This repository implements  
[**Eye-in-Finger: Smart Fingers for Delicate Assembly and Disassembly of LEGO**](https://arxiv.org/abs/2503.06848).

It detects and publishes LEGO brick offsets using an endoscope camera placed inside a LEGO manipulation End-Of-Arm Tool (EOAT).

## Instructions

### General Usage

#### Detect Offset and Visualize Augmented Camera View

```sh
python3 detect_offset.py
```

#### Publish Detected Offset and Tilt

- Publishes detected offset to `/tool_offset` and detected tilt to `/block_tilt`.

```sh
python3 pub_detect_offset.py
```

#### Use as a Visual Guide for Human Teleoperation

```sh
python3 detect_offset_manual.py
```

---

### MFI Setup - Publishing Tool Offset

#### On the Desktop Next to the Robots

1. **Find the Camera ID for Eye-in-Finger (EiF) for each robot:**

    ```sh
    python3 find_cam.py
    ```

    - This script visualizes all the cameras connected to the computer.
    - Note down the camera ID corresponding to the correct camera feed for each robot.
    - **Note:** `find_cam.py` does **not** work in headless mode.

2. **Publish the Camera Feed as a CompressedImage at the rostopic `/yk_{robot_name}/gen3_image`:**

    ```sh
    python3 pub_camera_feed.py robot_name camera_id
    ```

    - `robot_name` should **not** include the "yk_" prefix.
    - `camera_id` should be a number.

    Example:

    ```sh
    python3 pub_camera_feed.py destroyer 1
    ```

#### On the Desktop Running the Controller

3. **Receive the Camera Feed and Publish the Calculated Offset:**

    ```sh
    python3 compressed_pub_detect_offset.py destroyer
    ```

    - This publishes the offset (in the tool frame) to the topic `/yk_{robot_name}/tool_offset`.

---

### MFI Setup - Hardware Check, Calibration, and Height Adjustment

1. **Check Camera Position in EiF**

    - The bottom (the side with the wire) of the endoscope camera should be aligned flush with the rear (the side closer to the tool base) of the tool head of EiF.

    **Adjust Height:**
    - Loosen the screw on the tool head that's clamping down on the camera.
    - Move the camera to the correct position, then tighten the screw to lock it in place.

2. **Ensure Correct Camera Orientation**

    - Open the camera app and switch to the EiF camera feed.
    - Observe the occlusion caused by the tool head:
      - The side with one protrusion should appear on the right.
      - The side with two protrusions should also appear on the right.
    - They should line up mostly horizontally; small tilting (up to ±10 degrees) is acceptable.

    **Adjust Camera Orientation:**
    - Loosen the screw on the tool head that's clamping down on the camera.
    - Rotate the camera until the correct alignment is achieved, then tighten the screw to lock it.
    - It is recommended to hold the camera steady by the wire and rotate the tool head instead.

3. **Calibrate Zero Position**

    *(Do this if the offset detected by EiF is not close to zero when the tool is precisely lined up with the brick.)*

    1. Manually go to an accurate pick position.
    2. Move EiF height to $h_{peek}$.
    3. Adjust the global variable `TOOL_CENTER` in `detect_offset.py` until the detected offset is close to zero.

4. **Calibrate Offset Magnitude**

    *(Do this if the detected offset has the correct orientation but an incorrect magnitude, e.g., true offset = [1.0, -0.6], detected offset = [0.7, -0.4].)*

    1. Move EiF to $h_{peek}$, manually move the robot until the detected offset is zero.
    2. Jog the robot by $dx, dy$, record the detected offset by EiF $dex, dey$.
    3. Compute the error ratio of detection vs ground truth $er = \text{avg}(dex/dx, dey/dy)$.
    4. In `compute_offset()` in `detect_offset.py`, the default argument `z` is set to 30mm (this is just a number and does not reflect actual height). Replace `z` with `new_z = z * er`.
    5. Retry steps 1-2 to verify if the detected offset is now correct.

5. **Adjust Camera Distance to Brick When Computing Offset**

    *(Default peek height $h_{peek}$ is 17mm above pick-down height $h_{down}$. You should only need to perform this adjustment if a different camera with different focal characteristics is used.)*

    *(The segmentation model is trained for approximately 17mm above LEGO blocks. Using a different height may cause the model to malfunction.)*

    1. From $h_{down}$, move the camera upward until the image of the LEGO block is crisp and the model correctly segments the LEGO knobs. This can be checked by running:

    ```sh
    python3 detect_offset.py
    ```

    - Ensure that `show_yolo = True` when calling `compute_offset()`.

    2. Record the current height of the tool $h_{peek}$, calculate the difference $h_{diff} = h_{peek} - h_{down}$, and configure the robot to move to $h_{diff}$ above the LEGO block for offset detection before picking up.

    3. Part of the knob detection relies on the estimated knob size based on height. If the height changes (or a camera with a different focal length is used), you may notice that the green circles in the visualization are consistently larger or smaller than the actual knobs.

    - If this happens, go to the `cost_function()` in `detect_offset.py`, and modify `expected_radius` to match the estimated radius (in pixels) of the knobs at the new camera height.

    4. You will likely need to perform **Calibrate Zero Position** and **Calibrate Offset Magnitude** afterwards.

