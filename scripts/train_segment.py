from ultralytics import YOLO
from roboflow import Roboflow
rf = Roboflow(api_key="ysXcOkuwq46DKP58MBEg")
project = rf.workspace("test-5ev0m").project("studs-v4")
version = project.version(7)
dataset = version.download("yolov8")
model = YOLO("yolov8m-seg.pt")       
results = model.train(data="studs-v4-7/data.yaml", imgsz=640, batch=16, epochs=100, plots=True)      


                



                