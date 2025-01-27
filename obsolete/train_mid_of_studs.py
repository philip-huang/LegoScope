from roboflow import Roboflow
from ultralytics import YOLO
rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
project = rf.workspace("test-5ev0m").project("detect-mid-of-studs")
version = project.version(2)
dataset = version.download("yolov8")
                

model = YOLO("yolov8s.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)
save_dir = ("runs_locate")
# Train the model
results = model.train(data="Detect-mid-of-studs-2/data.yaml", imgsz=480, batch=12, epochs=20, plots=True, save_dir = save_dir)

