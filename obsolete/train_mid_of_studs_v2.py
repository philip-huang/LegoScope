from roboflow import Roboflow
from ultralytics import YOLO
rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
project = rf.workspace("test-5ev0m").project("mid_of_studs_v2")
version = project.version(3)
dataset = version.download("yolov8")
                

model = YOLO("yolov8n-pose.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)
save_dir = ("runs_locate")
# Train the model
results = model.train(data="mid_of_studs_v2-3/data.yaml", imgsz=640, batch=12, epochs=200, plots=True, save_dir = save_dir)

