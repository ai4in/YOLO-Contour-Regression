from ultralytics import YOLO

model=YOLO('/home/data2/mxl/ultralytics/ultralytics-main/runs/classify/train27/weights/last.pt')
for r in model.predict(source="/home/data2/mxl/ultralytics/ultralytics-main/dataset/xx_test/normal",save=True,conf=0.5,stream=True,imgsz=96):
    pass