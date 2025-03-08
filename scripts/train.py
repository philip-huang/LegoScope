from roboflow import Roboflow
from ultralytics import YOLO
rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
project = rf.workspace("test-5ev0m").project("light-ring-detection-v2")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO("yolov8s.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)
save_dir = ("runs")
# Train the model
results = model.train(data="light-ring-detection-v2-1/data.yaml", imgsz=480, batch=8, epochs=20, plots=True, save_dir = save_dir)