# Lego Scope

This repository provides software to detect and publish LEGO brick offset using an endoscope camera placed inside a LEGO manipulation End-Of-Arm Tool (EOAT).

## Instructions

### Detect Offset and Visualize Augmented Camera View
```sh
python3 detect_offset.py
```

### Publish Detected Offset and Tilt
This command publishes the detected offset to `/tool_offset` and the detected tilt to `/block_tilt`:
```sh
python3 pub_detect_offset.py
```

### Use as a Visual Guide for Human Teleoperation
```sh
python3 detect_offset_manual.py
